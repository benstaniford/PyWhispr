"""Application orchestrator: hotkey → record → transcribe → paste."""

from __future__ import annotations

import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from enum import Enum, auto
from importlib.resources import files

from PySide6.QtCore import QObject, QTimer, QUrl, Signal
from PySide6.QtMultimedia import QSoundEffect
from PySide6.QtWidgets import QApplication

from pywhispr.audio import AudioRecorder
from pywhispr.config import Config, load_config, save_config
from pywhispr.hotkey import create_hotkey_listener
from pywhispr.injector import TextInjector
from pywhispr.platform_setup import MACOS_PERMISSIONS_HELP, warn_if_missing_permissions
from pywhispr.stt import create_backend
from pywhispr.tray import TrayIcon
from pywhispr.ui.overlay import OverlayWindow

log = logging.getLogger(__name__)


class State(Enum):
    LOADING = auto()
    IDLE = auto()
    RECORDING = auto()
    TRANSCRIBING = auto()
    INSERTING = auto()


class PyWhisprApp(QObject):
    """Owns all components and the state machine. Lives on the Qt main thread.

    Cross-thread events (hotkey presses from pynput's thread, mic levels from
    the PortAudio callback thread, worker results) arrive as queued Qt signal
    emissions, so all state transitions happen on the main thread.
    """

    _hotkey_toggled = Signal()
    _mic_level = Signal(float)
    _model_ready = Signal()
    _model_failed = Signal(str)
    _transcribed = Signal(str)
    _transcribe_failed = Signal(str)

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.state = State.LOADING

        self.backend = create_backend(cfg)
        self.overlay = OverlayWindow()
        self.tray = TrayIcon(
            on_quit=self._quit,
            on_toggle=self._hotkey_toggled.emit,
            on_change_hotkey=self._change_hotkey,
        )
        self.injector = TextInjector(cfg.paste_delay_ms, cfg.clipboard_restore_delay_ms)
        self.recorder = AudioRecorder(device=cfg.input_device, on_level=self._mic_level.emit)
        self.listener = create_hotkey_listener(cfg.hotkey, on_toggle=self._hotkey_toggled.emit)
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pywhispr-stt")

        self._max_duration_timer = QTimer(self)
        self._max_duration_timer.setSingleShot(True)
        self._max_duration_timer.setInterval(cfg.max_recording_seconds * 1000)
        self._max_duration_timer.timeout.connect(self._on_max_duration)

        self._start_sound = self._load_sound("start.wav")
        self._stop_sound = self._load_sound("stop.wav")

        self._hotkey_toggled.connect(self._on_toggle)
        self._mic_level.connect(self.overlay.on_level)
        self._model_ready.connect(self._on_model_ready)
        self._model_failed.connect(self._on_model_failed)
        self._transcribed.connect(self._on_transcribed)
        self._transcribe_failed.connect(self._on_transcribe_failed)
        self.injector.finished.connect(self._on_insert_finished)

    # -- startup / shutdown ------------------------------------------------

    def start(self) -> None:
        self.tray.show()
        self.tray.set_status("Loading model…")
        if not warn_if_missing_permissions():
            self.tray.notify("PyWhispr: clipboard mode", MACOS_PERMISSIONS_HELP)
        try:
            self.listener.start()
        except Exception as exc:  # bad hotkey string, missing permission, ...
            self._fatal(f"Could not register hotkey {self.cfg.hotkey!r}: {exc}")
            return

        def load():
            try:
                self.backend.load()
                self._model_ready.emit()
            except Exception as exc:
                log.exception("Model load failed")
                self._model_failed.emit(str(exc))

        log.info("Loading %s (first run downloads the model, ~600 MB)...", self.backend.name)
        self._worker.submit(load)

    def _quit(self) -> None:
        self.listener.stop()
        if self.recorder.recording:
            self.recorder.stop()
        self._worker.shutdown(wait=False)
        QApplication.quit()

    def _fatal(self, message: str) -> None:
        log.error(message)
        self.tray.notify("PyWhispr error", message)
        QTimer.singleShot(8000, self._quit)

    # -- state transitions (main thread only) --------------------------------

    def _on_model_ready(self) -> None:
        self.state = State.IDLE
        self.tray.set_status(f"Ready — press {self.cfg.hotkey} to dictate")
        log.info("Ready. Press %s to start/stop dictation.", self.cfg.hotkey)

    def _on_model_failed(self, message: str) -> None:
        self._fatal(f"Model failed to load: {message}")

    def _on_toggle(self) -> None:
        if self.state == State.LOADING:
            self.overlay.show_status("Loading model…")
            QTimer.singleShot(1500, lambda: self.state == State.LOADING and self.overlay.hide_overlay())
        elif self.state == State.IDLE:
            self._start_recording()
        elif self.state == State.RECORDING:
            self._stop_recording()
        # TRANSCRIBING / INSERTING: ignore presses until the cycle completes

    def _start_recording(self) -> None:
        try:
            self.recorder.start()
        except Exception as exc:
            log.exception("Could not open microphone")
            self.tray.notify("Microphone error", str(exc))
            return
        self.state = State.RECORDING
        self._play(self._start_sound)
        self.overlay.show_recording()
        self.tray.set_status("Recording…", active=True)
        self._max_duration_timer.start()

    def _stop_recording(self) -> None:
        self._max_duration_timer.stop()
        audio = self.recorder.stop()
        self._play(self._stop_sound)
        self.state = State.TRANSCRIBING
        self.overlay.show_status("Transcribing…")
        self.tray.set_status("Transcribing…")

        def transcribe():
            try:
                self._transcribed.emit(self.backend.transcribe(audio))
            except Exception as exc:
                log.exception("Transcription failed")
                self._transcribe_failed.emit(str(exc))

        self._worker.submit(transcribe)

    def _on_max_duration(self) -> None:
        if self.state == State.RECORDING:
            log.info("Max recording duration reached, stopping")
            self._stop_recording()

    def _on_transcribed(self, text: str) -> None:
        if not text.strip():
            log.info("Empty transcription, nothing to insert")
            self._finish_cycle()
            return
        log.info("Transcribed %d characters", len(text))
        self.state = State.INSERTING
        self.overlay.hide_overlay()
        self.injector.insert(text)

    def _on_transcribe_failed(self, message: str) -> None:
        self.tray.notify("Transcription failed", message)
        self._finish_cycle()

    def _on_insert_finished(self, auto_pasted: bool) -> None:
        if not auto_pasted:
            paste_key = "Cmd+V" if sys.platform == "darwin" else "Ctrl+V"
            self.tray.notify(
                "Copied to clipboard", f"Press {paste_key} to paste your dictation."
            )
        self._finish_cycle()

    def _finish_cycle(self) -> None:
        self.overlay.hide_overlay()
        self.state = State.IDLE
        self.tray.set_status(f"Ready — press {self.cfg.hotkey} to dictate")

    def _change_hotkey(self) -> None:
        """Tray menu: capture a new chord, save it, and re-register the listener."""
        if self.state not in (State.IDLE, State.LOADING):
            return
        from pywhispr.ui.hotkey_dialog import HotkeyCaptureDialog

        # Stop listening while the dialog is up so pressing the current chord
        # inside it doesn't start a recording.
        self.listener.stop()
        new_chord = HotkeyCaptureDialog.capture(self.cfg.hotkey)

        if new_chord and new_chord != self.cfg.hotkey:
            old_chord = self.cfg.hotkey
            try:
                self.listener = create_hotkey_listener(new_chord, self._hotkey_toggled.emit)
                self.listener.start()
                self.cfg.hotkey = new_chord
                save_config(self.cfg)
                log.info("Hotkey changed to %s", new_chord)
            except Exception as exc:
                log.exception("Could not register new hotkey")
                self.tray.notify("Hotkey not changed", f"Could not register {new_chord!r}: {exc}")
                self.listener = create_hotkey_listener(old_chord, self._hotkey_toggled.emit)
                self.listener.start()
        else:
            self.listener.start()

        if self.state != State.LOADING:
            self.tray.set_status(f"Ready — press {self.cfg.hotkey} to dictate")

    # -- sounds --------------------------------------------------------------

    def _load_sound(self, name: str) -> QSoundEffect | None:
        if not self.cfg.play_sounds:
            return None
        effect = QSoundEffect(self)
        effect.setSource(QUrl.fromLocalFile(str(files("pywhispr") / "assets" / name)))
        effect.setVolume(0.4)
        return effect

    def _play(self, effect: QSoundEffect | None) -> None:
        if effect is not None:
            effect.play()


def run_app() -> int:
    from PySide6.QtGui import QIcon

    from pywhispr.tray import app_pixmap

    app = QApplication(sys.argv)
    app.setApplicationName("PyWhispr")
    app.setWindowIcon(QIcon(app_pixmap()))
    app.setQuitOnLastWindowClosed(False)  # tray app: no windows most of the time

    whispr = PyWhisprApp(load_config())
    whispr.start()
    return app.exec()
