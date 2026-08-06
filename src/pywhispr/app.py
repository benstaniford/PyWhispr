"""Application orchestrator: hotkey → record → transcribe → paste."""

from __future__ import annotations

import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum, auto
from importlib.resources import files

from PySide6.QtCore import QObject, QTimer, QUrl, Signal
from PySide6.QtMultimedia import QSoundEffect
from PySide6.QtWidgets import QApplication

from pywhispr import perf
from pywhispr.api import QUEUE_TIMEOUT_SECONDS, TranscriptionServer
from pywhispr.audio import AudioRecorder
from pywhispr.caret import ContextTracker
from pywhispr.config import Config, save_config
from pywhispr.ducking import create_ducker
from pywhispr.filler import filler_words, is_deletion_only, remove_fillers
from pywhispr.hotkey import create_hotkey_listener
from pywhispr.injector import TextInjector
from pywhispr.join import join_text
from pywhispr.logging_setup import log_environment, log_path
from pywhispr.platform_setup import MACOS_PERMISSIONS_HELP, warn_if_missing_permissions
from pywhispr.stt import create_backend
from pywhispr.stt.base import SAMPLE_RATE
from pywhispr.tray import TrayIcon
from pywhispr.ui.overlay import OverlayWindow
from pywhispr.vocab import Rule, apply_vocabulary, load_vocabulary

log = logging.getLogger(__name__)

# Holding a double-tap's second key at least this long makes it push-to-talk
# (stop on release); a quicker double-tap latches recording instead.
PUSH_TO_TALK_HOLD_SECONDS = 0.35


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
    _hotkey_released = Signal(float)  # double-tap activating key released; held seconds
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
        log.debug("Backend selected: %s", self.backend.name)
        self.overlay = OverlayWindow()
        self.tray = TrayIcon(
            on_quit=self._quit,
            on_toggle=self._hotkey_toggled.emit,
            on_change_hotkey=self._change_hotkey,
            on_edit_vocabulary=self._edit_vocabulary,
            on_enable_gpu=self._enable_gpu,
        )
        self._progress_window = None  # kept alive while anything is downloading
        self._load_model = None  # set by start(), possibly deferred behind GPU setup
        self._asked_about_gpu = False
        self._waiting_for_gpu_setup = False
        # Rebound wholesale (never mutated) when the editor saves, so the API's
        # request threads can read it without a lock.
        self._vocab: list[Rule] = load_vocabulary()
        self._fillers = filler_words(cfg.extra_filler_words, cfg.keep_filler_words)
        self.injector = TextInjector(cfg.paste_delay_ms, cfg.clipboard_restore_delay_ms)
        self._context = ContextTracker(
            max_chars=cfg.context_chars, memory_seconds=cfg.context_memory_seconds
        )
        self._last_inserted: str | None = None
        self.recorder = AudioRecorder(device=cfg.input_device, on_level=self._mic_level.emit)
        self.ducker = create_ducker(cfg)
        self.listener = create_hotkey_listener(
            cfg.hotkey,
            on_toggle=self._hotkey_toggled.emit,
            on_release=self._hotkey_released.emit,
        )
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pywhispr-stt")
        self._model_error: str | None = None

        # Network API. Requests run on their own threads but hand the actual
        # transcription to _worker, so the model is still only ever used by one
        # thread at a time and remote work queues behind local dictation.
        self.api = (
            TranscriptionServer(
                transcribe=self._api_transcribe,
                status=self._api_status,
                host=cfg.api_host,
                port=cfg.api_port,
                max_audio_seconds=cfg.api_max_audio_seconds,
                max_queue=cfg.api_max_queue,
            )
            if cfg.api_enabled
            else None
        )

        # Push-to-talk bookkeeping: did the last activation start a recording,
        # and when. Set on activate, consumed on the activating key's release.
        self._activation_started_recording = False

        self._max_duration_timer = QTimer(self)
        self._max_duration_timer.setSingleShot(True)
        self._max_duration_timer.setInterval(cfg.max_recording_seconds * 1000)
        self._max_duration_timer.timeout.connect(self._on_max_duration)

        self._start_sound = self._load_sound("start.wav")
        self._stop_sound = self._load_sound("stop.wav")

        self._hotkey_toggled.connect(self._on_activate)
        self._hotkey_released.connect(self._on_activation_key_released)
        self._mic_level.connect(self.overlay.on_level)
        self._model_ready.connect(self._on_model_ready)
        self._model_failed.connect(self._on_model_failed)
        self._transcribed.connect(self._on_transcribed)
        self._transcribe_failed.connect(self._on_transcribe_failed)
        self.injector.finished.connect(self._on_insert_finished)

        perf.install_loop_monitor(self)

    # -- startup / shutdown ------------------------------------------------

    def start(self) -> None:
        # Without importing onnxruntime: the GPU question has not been asked yet,
        # and DirectML can only replace onnxruntime before its first import. The
        # backend logs the version and providers when it loads.
        log_environment(allow_onnxruntime_import=False)
        log.info(
            "Starting: hotkey=%s, device=%s, api=%s, max_recording=%ss",
            self.cfg.hotkey,
            self.cfg.input_device if self.cfg.input_device is not None else "system default",
            f"{self.cfg.api_host}:{self.cfg.api_port}" if self.cfg.api_enabled else "disabled",
            self.cfg.max_recording_seconds,
        )
        self.tray.show()
        self.tray.set_status("Loading model…")
        if self.api is not None and not self.api.start():
            self.tray.notify(
                "PyWhispr: network API off",
                f"Port {self.cfg.api_port} is already in use. Dictation still works.",
            )
        if not warn_if_missing_permissions():
            self.tray.notify("PyWhispr: clipboard mode", MACOS_PERMISSIONS_HELP)
        try:
            self.listener.start()
            log.info("Hotkey listener started (%s)", type(self.listener).__name__)
        except Exception as exc:  # bad hotkey string, missing permission, ...
            # Not fatal: the tray menu can still change the hotkey or quit, and
            # an app that silently disappears teaches the user nothing.
            log.exception("Could not register hotkey %r", self.cfg.hotkey)
            self._report_error(
                "Hotkey not registered",
                f"Could not register {self.cfg.hotkey!r}: {exc}. "
                "Use the tray menu to pick a different one.",
            )
        self._check_double_tap_permission(self.cfg.hotkey)

        def load():
            started = time.monotonic()
            try:
                self.backend.load()
                log.info("Model ready in %.1fs", time.monotonic() - started)
                self._model_ready.emit()
            except Exception as exc:
                log.exception("Model load failed after %.1fs", time.monotonic() - started)
                self._model_failed.emit(f"{type(exc).__name__}: {exc}")

        self._load_model = load
        self._offer_another_drive()
        if self._offer_gpu_before_downloading():
            return  # the model load waits for the CUDA setup to finish
        self._begin_model_load()

    def _begin_model_load(self) -> None:
        log.info("Loading %s (a first run downloads the model)...", self.backend.name)
        self._show_model_download()
        self._worker.submit(self._load_model)

    def _offer_another_drive(self) -> None:
        """Ask where the downloads should go, while asking still changes anything.

        Before the GPU question, because it decides where the CUDA libraries land
        too, and before any load, because afterwards the bytes are already on the
        wrong drive. Silent unless a location is already configured or the default
        one has room — see storage_dialog.should_ask.
        """
        if self.cfg.model_cache_dir or self.cfg.cuda_dir:
            return
        from pywhispr.download import model_cached

        if model_cached():
            return
        from pywhispr.ui.storage_dialog import ask_where_to_store, should_ask

        if not should_ask():
            return
        base = ask_where_to_store()
        if base is None:
            return

        from pywhispr.storage import apply_overrides, set_base_dir

        set_base_dir(self.cfg, base)
        save_config(self.cfg)
        # The environment alone is not enough here: asking the question imported
        # huggingface_hub.constants to find the default, so its HF_HUB_CACHE is
        # already fixed at the old path.
        apply_overrides(self.cfg)
        self._redirect_hf_cache()

    def _redirect_hf_cache(self) -> None:
        """Make the already-imported huggingface_hub honour the new directory.

        Patching the constant is enough because only ``constants`` has been imported
        so far: the modules that copy the value out of it come in later, with
        ``onnx_asr`` inside the model load. Failing is not fatal either way — the
        choice is saved, so a restart picks it up.
        """
        target = self.cfg.model_cache_dir
        if not target:
            return
        try:
            import huggingface_hub.constants as hf

            hf.HF_HUB_CACHE = target
        except Exception:
            log.debug("Could not redirect the live model cache path", exc_info=True)

    def _offer_gpu_before_downloading(self) -> bool:
        """Ask about the GPU first, while the choice still saves a download.

        Asked after loading — as it used to be — the answer arrives too late: the
        quantised weights the GPU has no use for are already on disk. True means the
        setup is running and will start the model load when it is done.
        """
        from pywhispr.download import model_cached

        if model_cached() or not self.cfg.offer_gpu_setup:
            return False
        kind = self._acceleration_on_offer()
        if kind is None:
            return False

        from pywhispr.ui.setup_window import ask_to_enable

        self._asked_about_gpu = True
        answer = ask_to_enable(first_run=True, kind=kind)
        if answer is None:
            self.cfg.offer_gpu_setup = False
            save_config(self.cfg)
            log.info("GPU acceleration declined for good")
        if not answer:
            return False

        if kind == "cuda":
            # Full precision from here on: the GPU is slower on the quantised
            # weights, so fetching them as well would be the waste this ordering
            # avoids. DirectML keeps whatever is configured — see setup_window.
            self.cfg.model_quantization = ""
            save_config(self.cfg)
            self.backend = create_backend(self.cfg)
        self._waiting_for_gpu_setup = True
        self._run_gpu_setup(first_run=True, kind=kind)
        return True

    def _acceleration_on_offer(self) -> str | None:
        """"cuda", "directml", or None — which GPU path is worth offering here.

        CUDA first because it is much faster where it runs at all; DirectML is the
        fallback for what CUDA 13 dropped (pre-Turing NVIDIA) and never covered
        (AMD, Intel).
        """
        from pywhispr import cuda, directml

        worth_it, why_not = cuda.can_offer()
        if worth_it:
            return "cuda"
        log.debug("Not offering CUDA: %s", why_not)

        worth_it, why_not_dml = directml.can_offer()
        if worth_it:
            return "directml"
        log.debug("Not offering DirectML either: %s", why_not_dml)
        return None

    def _run_gpu_setup(self, first_run: bool = False, kind: str = "cuda"):
        window = self._setup_window()
        window.setup_finished.connect(self._on_gpu_setup_finished)
        window.start_gpu_setup(first_run=first_run, kind=kind)
        return window

    def _setup_window(self):
        """The single window everything downloading reports into.

        One window with a line per activity, because either download can start while
        the other is running — the model at startup, GPU setup from the tray — and a
        window each meant two bars counting overlapping bytes.
        """
        from pywhispr.ui.foreground import show_in_front
        from pywhispr.ui.setup_window import SetupWindow

        if self._progress_window is None:
            self._progress_window = SetupWindow()
            self._progress_window.finished.connect(self._on_setup_window_closed)
        show_in_front(self._progress_window)
        return self._progress_window

    def _on_setup_window_closed(self, _result: int) -> None:
        self._progress_window = None

    def _on_gpu_setup_finished(self, worked: bool) -> None:
        """The libraries are in place (or are not), so the model can load now.

        Loading in this process rather than after a restart is deliberate:
        onnxruntime resolves providers when a session is built, and the libraries
        arrived before that happened.
        """
        if not self._waiting_for_gpu_setup:
            return  # tray-triggered setup: the model is already loaded
        self._waiting_for_gpu_setup = False
        if not worked:
            self.cfg.model_quantization = None
            save_config(self.cfg)
            self.backend = create_backend(self.cfg)
            log.info("GPU setup did not work out; loading the CPU model instead")
        else:
            self._activate_directml_if_just_installed()
        self._begin_model_load()

    def _activate_directml_if_just_installed(self) -> None:
        """Swap in the DirectML onnxruntime now, while that is still possible.

        Nothing has imported onnxruntime yet on this path — the startup report is
        asked not to — so the download that just finished can take effect in this
        process. Miss this moment and it only applies after a restart, which is
        what the first run used to do while promising nothing.
        """
        from pywhispr import directml

        if not directml.is_installed() or directml.is_active():
            return
        if directml.activate():
            self.backend = create_backend(self.cfg)  # a fresh backend picks the variant again

    def _show_model_download(self) -> None:
        """On a first run, show the download rather than a silent "Loading…"."""
        from pywhispr.download import model_cached

        # Before the cached check, not after: download_mb depends on which variant
        # will be fetched, and load() does not decide until it runs on the worker
        # thread — so the size shown here would be the full-precision one whatever
        # we were about to download.
        choose = getattr(self.backend, "choose_quantization", None)
        if choose is not None:
            try:
                choose()
            except Exception:
                log.debug("Could not pick the model variant early", exc_info=True)

        # Against the size of *this* variant. A flat 400 MB meant that switching
        # from int8 to full precision — enabling a GPU does exactly that — saw
        # 785 MB in the cache, called it cached, and fetched 2.4 GB behind a
        # motionless "Loading model…". One such fetch died after 202 seconds with
        # nothing on screen to say so.
        expected_mb = getattr(self.backend, "download_mb", None)
        # A backend is duck-typed here, so the size is only trusted when it really
        # is one; anything else falls back to the old flat threshold rather than
        # throwing from the middle of startup.
        minimum_mb = int(expected_mb * 0.8) if isinstance(expected_mb, (int, float)) else 400
        if model_cached(minimum_mb=minimum_mb):
            return

        self._setup_window().track_model_download(self.backend.download_mb)

    def _finish_model_download(self, message: str | None = None) -> None:
        if self._progress_window is not None:
            self._progress_window.finish_model(message)

    def _quit(self) -> None:
        log.info("Quitting")
        self.listener.stop()
        if self.api is not None:
            self.api.stop()  # before the worker, so no request is left orphaned
        try:
            if self.recorder.recording:
                self.recorder.stop()
        finally:
            # Unconditional, and shielded from a failing recorder stop: Windows
            # remembers per-app mixer levels, so quitting while ducked would
            # leave the user's other apps quiet for good.
            self.ducker.restore()
        self._worker.shutdown(wait=False)
        QApplication.quit()

    def _report_error(self, title: str, message: str) -> None:
        """Surface a non-recoverable-but-not-fatal error without exiting.

        Tray balloons are easy to miss (Windows routes them straight to the
        notification centre), so the message also lands in the tray tooltip,
        in the log, and on the overlay the next time the hotkey is pressed.
        """
        log.error("%s: %s", title, message)
        self.tray.set_status(message)
        self.tray.notify(title, f"{message}\n\nDetails: {log_path()}")

    # -- network API hooks (called on API request threads) --------------------

    def _api_status(self) -> dict:
        """Snapshot for /v1/health.

        self.state is only ever written on the main thread, so an unsynchronised
        read here can at worst be one instant stale — harmless, since the
        backend itself rejects use before it is loaded.
        """
        if self._model_error is not None:
            status = "error"
        elif self.state == State.LOADING:
            status = "loading"
        else:
            status = "ready"
        return {"status": status, "backend": self.backend.name, "error": self._model_error}

    def _api_transcribe(self, audio) -> str:
        """Run a remote request on the single STT worker and wait for the text.

        No continuation joining: that belongs to the local dictation cycle,
        which has a caret to join onto. A remote caller has neither that nor
        any session. Filler removal and the vocabulary do apply, because they
        are standing preferences about the text rather than session state.
        """
        text = self._worker.submit(self.backend.transcribe, audio).result(
            timeout=QUEUE_TIMEOUT_SECONDS + len(audio) / SAMPLE_RATE
        )
        return self._corrected(self._cleaned(text))

    # -- state transitions (main thread only) --------------------------------

    def _set_state(self, state: State) -> None:
        """Single funnel for state changes, so the log shows the machine move.

        A stuck state is the symptom of nearly every "nothing happened" bug, so
        every transition is recorded with where it came from.
        """
        if state is not self.state:
            log.debug("State %s → %s", self.state.name, state.name)
        self.state = state

    def _on_model_ready(self) -> None:
        self._finish_model_download()
        self._set_state(State.IDLE)
        self.tray.set_status(f"Ready — press {self.cfg.hotkey} to dictate")
        log.info("Ready. Press %s to start/stop dictation.", self.cfg.hotkey)
        QTimer.singleShot(0, self._maybe_offer_gpu)

    # -- GPU acceleration ----------------------------------------------------

    def _maybe_offer_gpu(self) -> None:
        """Offer the CUDA download once, if this machine would benefit.

        This is the path for an existing install, where the model is already
        downloaded and there is nothing left to save by asking first.
        """
        from pywhispr.stt.onnx_backend import session_providers

        if not self.cfg.offer_gpu_setup or self._asked_about_gpu:
            return
        providers = session_providers(getattr(self.backend, "_model", None))
        if any(p != "CPUExecutionProvider" for p in providers):
            return  # already accelerated
        if self._acceleration_on_offer() is None:
            return
        self._enable_gpu(asked_by_user=False)

    def _enable_gpu(self, asked_by_user: bool = True) -> None:
        from pywhispr import cuda, directml
        from pywhispr.ui.foreground import show_in_front
        from pywhispr.ui.setup_window import ask_to_enable

        if self._progress_window is not None and self._progress_window.gpu_running:
            show_in_front(self._progress_window)  # already doing it
            return
        kind = self._acceleration_on_offer()
        if kind is None:
            already = cuda.is_installed() or directml.is_installed()
            if asked_by_user and not already:
                self.tray.notify("GPU acceleration not available", cuda.can_offer()[1])
                return
            kind = "directml" if directml.is_installed() else "cuda"

        answer = ask_to_enable(kind=kind)
        if answer is None:
            self.cfg.offer_gpu_setup = False
            save_config(self.cfg)
            log.info("GPU acceleration declined for good")
            return
        if not answer:
            return
        # Reuses the window if the model is still downloading into it.
        self._run_gpu_setup(kind=kind)

    def _on_model_failed(self, message: str) -> None:
        """Model load failed: stay alive and keep saying so.

        The app used to quit 8 seconds later, which looked identical to a
        crash — the tray icon just vanished and no window was ever shown. Now
        it stays in LOADING with an error recorded, so the tooltip, the
        overlay, /v1/health and the log all name the cause.
        """
        self._model_error = message
        self._finish_model_download(f"The model could not be loaded.\n\n{message}")
        self._report_error("Model failed to load", message)

    def _on_activate(self) -> None:
        """Hotkey pressed (chord, or the second tap of a double-tap)."""
        log.debug("Hotkey activated in state %s", self.state.name)
        self._activation_started_recording = self.state == State.IDLE
        self._on_toggle()

    def _on_activation_key_released(self, held_seconds: float) -> None:
        """Double-tap activating key released.

        If that activation started a recording and the key was *held* (rather
        than quickly double-tapped), stop on release — push-to-talk. A quick
        double-tap instead leaves recording latched until the next double-tap.
        Querying self.state keeps this correct if the max-duration guard or an
        error already ended the recording.
        """
        if (
            self._activation_started_recording
            and self.state == State.RECORDING
            and held_seconds >= PUSH_TO_TALK_HOLD_SECONDS
        ):
            self._stop_recording()
        self._activation_started_recording = False

    def _on_toggle(self) -> None:
        if self.state == State.LOADING:
            # Failed loads stay in LOADING, so say which of the two it is.
            message = "Model failed — see log" if self._model_error else "Loading model…"
            self.overlay.show_status(message)
            QTimer.singleShot(1500, lambda: self.state == State.LOADING and self.overlay.hide_overlay())
            if self._model_error:
                log.warning("Hotkey pressed but the model never loaded: %s", self._model_error)
        elif self.state == State.IDLE:
            self._start_recording()
        elif self.state == State.RECORDING:
            self._stop_recording()
        else:
            log.debug("Hotkey ignored: still %s", self.state.name)

    def _start_recording(self) -> None:
        try:
            self.recorder.start()
        except Exception as exc:
            log.exception("Could not open microphone")
            self.tray.notify("Microphone error", str(exc))
            return
        self._set_state(State.RECORDING)
        self.ducker.duck()
        self._play(self._start_sound)
        self.overlay.show_recording()
        self.tray.set_status("Recording…", active=True)
        self._max_duration_timer.start()

    def _stop_recording(self) -> None:
        self._max_duration_timer.stop()
        try:
            audio = self.recorder.stop()
        finally:
            # Even when PortAudio fails to close the stream, the other apps
            # must get their volume back — Windows would remember the ducked
            # levels forever.
            self.ducker.restore()
        self._play(self._stop_sound)
        self._set_state(State.TRANSCRIBING)
        self.overlay.show_status("Transcribing…")
        self.tray.set_status("Transcribing…")
        log.info("Transcribing %.1fs of audio", len(audio) / SAMPLE_RATE)

        def transcribe():
            started = time.monotonic()
            try:
                text = self.backend.transcribe(audio)
                log.info("Transcription took %.1fs", time.monotonic() - started)
                # From here to the text landing in the document is the part the
                # user sees as "the pill vanished but nothing was pasted yet".
                perf.begin("post-transcribe")
                self._transcribed.emit(text)
            except Exception as exc:
                log.exception("Transcription failed")
                self._transcribe_failed.emit(f"{type(exc).__name__}: {exc}")

        self._worker.submit(transcribe)

    def _on_max_duration(self) -> None:
        if self.state == State.RECORDING:
            log.info("Max recording duration reached, stopping")
            self._stop_recording()

    def _on_transcribed(self, text: str) -> None:
        perf.mark("signal-delivered")
        # Before the empty check: a recording of nothing but "um" leaves nothing.
        text = self._cleaned(text)
        perf.mark("fillers")
        if not text.strip():
            log.info("Empty transcription, nothing to insert")
            self._finish_cycle()
            return
        log.info("Transcribed %d characters", len(text))
        self._set_state(State.INSERTING)
        self.overlay.hide_overlay()
        perf.mark("overlay-hidden")
        # Vocabulary next: it can change the opening word, which is the word
        # the join then decides about.
        corrected = self._corrected(text)
        perf.mark("vocabulary")
        self._last_inserted = self._joined(corrected)
        perf.mark("join")
        self.injector.insert(self._last_inserted)

    def _cleaned(self, text: str) -> str:
        """Take the hesitations out, degrading to the raw transcript on any doubt.

        Wrapped like _corrected and _joined: the audio is gone by now. The
        tripwire is remove_fillers' own contract, deletions only.
        """
        if not text or not self.cfg.remove_fillers or not self._fillers:
            return text
        try:
            cleaned = remove_fillers(text, self._fillers)
        except Exception:
            log.exception("Filler removal failed; using the raw transcript")
            return text
        if not is_deletion_only(text, cleaned):
            log.error(
                "Filler removal added text (%d characters from %d); using the raw transcript",
                len(cleaned),
                len(text),
            )
            return text
        return cleaned

    def _corrected(self, text: str) -> str:
        """Apply the user's vocabulary to a finished transcript.

        Wrapped for the same reason _joined is: by now the audio is gone, so a
        bug in here has to degrade to the raw transcript rather than lose it.
        """
        if not text or not self.cfg.vocabulary_enabled or not self._vocab:
            return text
        try:
            corrected = apply_vocabulary(text, self._vocab, fuzzy=self.cfg.vocabulary_fuzzy)
        except Exception:
            log.exception("Vocabulary pass failed; using the raw transcript")
            return text
        # A tripwire, not a proof: corrections rewrite words, so unlike the join
        # we cannot check the text is untouched — only that none of it ran away.
        if not corrected.strip() or not 0.5 <= len(corrected) / len(text) <= 2:
            log.error(
                "Vocabulary produced %d characters from %d; using the raw transcript",
                len(corrected),
                len(text),
            )
            return text
        return corrected

    def _joined(self, text: str) -> str:
        """Adapt the transcript to what precedes the caret.

        Wrapped whole, because losing a transcript is unrecoverable — the audio
        is already gone. Any failure in here, including one inside join_text,
        falls back to the raw text rather than propagating: an exception
        escaping _on_transcribed would strand the app in INSERTING, where
        injector.finished never fires and every hotkey is ignored.
        """
        if not self.cfg.join_continuations:
            return text
        try:
            preceding = self._context.preceding_text()
            joined = join_text(
                preceding, text, lowercase_continuations=self.cfg.lowercase_continuations
            )
        except Exception:
            log.exception("Continuation join failed; inserting verbatim")
            return text
        # join_text promises "at most one extra leading character, everything
        # from text[1] onward untouched". Checking it here means a bug in there
        # can only ever degrade to today's behaviour.
        if not (joined.endswith(text[1:]) and 0 <= len(joined) - len(text) <= 1):
            log.error("Join produced unexpected output (%d chars); inserting verbatim", len(joined))
            return text
        return joined

    def _on_transcribe_failed(self, message: str) -> None:
        log.error("Transcription failed: %s", message)
        self.tray.notify("Transcription failed", message)
        self._context.invalidate()
        self._finish_cycle()

    def _on_insert_finished(self, auto_pasted: bool) -> None:
        log.debug("Insertion finished (auto_pasted=%s)", auto_pasted)
        if auto_pasted and self._last_inserted is not None:
            self._context.remember(self._last_inserted)
        else:
            # Clipboard-only mode: nothing entered the document, so there is no
            # caret to join onto next time — and whatever we remembered before
            # is no longer what the caret sits after either.
            self._context.invalidate()
        if not auto_pasted:
            paste_key = "Cmd+V" if sys.platform == "darwin" else "Ctrl+V"
            self.tray.notify(
                "Copied to clipboard", f"Press {paste_key} to paste your dictation."
            )
        self._finish_cycle()

    def _finish_cycle(self) -> None:
        perf.mark("cycle-finished")
        perf.end()
        self._last_inserted = None
        self.overlay.hide_overlay()
        self._set_state(State.IDLE)
        self.tray.set_status(f"Ready — press {self.cfg.hotkey} to dictate")

    def _change_hotkey(self) -> None:
        """Tray menu: capture a new chord, save it, and re-register the listener."""
        if self.state not in (State.IDLE, State.LOADING):
            return
        from pywhispr.ui.hotkey_dialog import HotkeyCaptureDialog

        # The dialog takes focus, so whatever we remembered inserting is no
        # longer behind the caret.
        self._context.invalidate()
        # Stop listening while the dialog is up so pressing the current chord
        # inside it doesn't start a recording.
        self.listener.stop()
        new_chord = HotkeyCaptureDialog.capture(self.cfg.hotkey)

        if new_chord and new_chord != self.cfg.hotkey:
            old_chord = self.cfg.hotkey
            try:
                self.listener = create_hotkey_listener(
                    new_chord, self._hotkey_toggled.emit, self._hotkey_released.emit
                )
                self.listener.start()
                self.cfg.hotkey = new_chord
                save_config(self.cfg)
                log.info("Hotkey changed to %s", new_chord)
                self._check_double_tap_permission(new_chord)
            except Exception as exc:
                log.exception("Could not register new hotkey")
                self.tray.notify("Hotkey not changed", f"Could not register {new_chord!r}: {exc}")
                self.listener = create_hotkey_listener(
                    old_chord, self._hotkey_toggled.emit, self._hotkey_released.emit
                )
                self.listener.start()
        else:
            self.listener.start()

        if self.state != State.LOADING:
            self.tray.set_status(f"Ready — press {self.cfg.hotkey} to dictate")

    def _edit_vocabulary(self) -> None:
        """Tray menu: edit the custom vocabulary and apply it without a restart."""
        if self.state not in (State.IDLE, State.LOADING):
            return
        from pywhispr.ui.vocab_dialog import VocabularyDialog
        from pywhispr.vocab import (
            TEMPLATE,
            load_vocabulary_text,
            parse_vocabulary,
            save_vocabulary_text,
        )

        # The dialog takes focus, so what we remembered inserting is no longer
        # behind the caret; and a dictation pasted into the editor helps nobody.
        self._context.invalidate()
        self.listener.stop()
        try:
            edited = VocabularyDialog.edit(load_vocabulary_text() or TEMPLATE)
            if edited is not None:
                save_vocabulary_text(edited)
                self._vocab = parse_vocabulary(edited)
                log.info("Vocabulary updated: %d term(s)", len(self._vocab))
        except Exception as exc:
            # Losing the edit is annoying; losing the tray app is worse.
            log.exception("Could not save the vocabulary")
            self.tray.notify("Vocabulary not saved", str(exc))
        finally:
            try:
                self.listener.start()
            except Exception:
                log.exception("Could not restart the hotkey listener after editing vocabulary")

    def _check_double_tap_permission(self, chord: str) -> None:
        """Double-tap hotkeys need Input Monitoring on macOS; guide the user."""
        from pywhispr.hotkey import DOUBLE_TAP_PREFIX
        from pywhispr.platform_setup import (
            MACOS_INPUT_MONITORING_HELP,
            check_macos_input_monitoring,
            request_macos_input_monitoring,
        )

        if not chord.startswith(DOUBLE_TAP_PREFIX):
            return
        if not check_macos_input_monitoring():
            request_macos_input_monitoring()
            log.warning(MACOS_INPUT_MONITORING_HELP)
            self.tray.notify("Input Monitoring needed", MACOS_INPUT_MONITORING_HELP)

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

    from pywhispr.logging_setup import install_qt_message_handler
    from pywhispr.startup import prepare
    from pywhispr.tray import app_pixmap

    # Here rather than only in cli.main: the frozen executable with no arguments
    # comes straight here, so this is the one place every path to the app passes
    # through. Idempotent, so arriving via cli.main is not a problem.
    config = prepare("run")

    install_qt_message_handler()  # before QApplication, to catch platform-plugin gripes
    app = QApplication(sys.argv)
    app.setApplicationName("PyWhispr")
    app.setWindowIcon(QIcon(app_pixmap()))
    app.setQuitOnLastWindowClosed(False)  # tray app: no windows most of the time

    whispr = PyWhisprApp(config)
    whispr.start()
    return app.exec()
