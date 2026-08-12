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

from pywhispr import flavor, gpu
from pywhispr.api import QUEUE_TIMEOUT_SECONDS, TranscriptionServer
from pywhispr.audio import AudioRecorder
from pywhispr.caret import ContextTracker
from pywhispr.config import Config, save_config
from pywhispr.ducking import create_ducker
from pywhispr.filler import filler_words, is_deletion_only, remove_fillers
from pywhispr.history import TranscriptHistory
from pywhispr.hotkey import create_hotkey_listener
from pywhispr.injector import TextInjector
from pywhispr.join import join_text
from pywhispr.logging_setup import log_environment, log_path
from pywhispr.platform_setup import MACOS_PERMISSIONS_HELP, warn_if_missing_permissions
from pywhispr.plugins.actions import ActionRunner
from pywhispr.plugins.engine import PendingAction, apply_plugins
from pywhispr.plugins.registry import load_plugins
from pywhispr.scratch import compile_reset_phrases, is_suffix_of, strip_before_reset
from pywhispr.stt import create_backend
from pywhispr.stt.base import SAMPLE_RATE
from pywhispr.tray import TrayIcon
from pywhispr.ui.overlay import OverlayWindow
from pywhispr.vocab import Rule, apply_vocabulary, load_vocabulary

log = logging.getLogger(__name__)

# Holding a double-tap's second key at least this long makes it push-to-talk
# (stop on release); a quicker double-tap latches recording instead.
PUSH_TO_TALK_HOLD_SECONDS = 0.35

# After the history picker closes, how long the previously focused window is
# given to take the focus back before the paste keystroke goes out.
FOCUS_RESTORE_MS = 150

# How long "Starting over…" replaces the waveform after the reset hotkey.
RESET_FEEDBACK_MS = 700


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
    _reset_requested = Signal()
    _history_requested = Signal()
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
        # No GPU entry on macOS: neither CUDA nor DirectML has a build for it and
        # Apple Silicon is on the Metal GPU through MLX already, so the entry could
        # only ever say no — and until the triggered(checked) fix in tray.py it said
        # yes instead and started a download that could only fail.
        gpu_possible = gpu.supported()
        self.tray = TrayIcon(
            on_quit=self._quit,
            on_toggle=self._hotkey_toggled.emit,
            on_change_hotkey=self._change_hotkey,
            on_edit_vocabulary=self._edit_vocabulary,
            on_open_plugins=self._open_plugins_folder if cfg.plugins_enabled else None,
            on_enable_gpu=self._enable_gpu if gpu_possible else None,
            on_disable_gpu=self._disable_gpu if gpu_possible else None,
            gpu_active=self._gpu_active if gpu_possible else None,
            on_show_history=self._history_requested.emit,
            # Lite only: the model runs on another machine, so the user needs a
            # way to say which one. The full app never shows this entry.
            on_set_server=self._set_server if flavor.IS_LITE else None,
        )
        self._progress_window = None  # kept alive while anything is downloading
        self._load_model = None  # set by start(), possibly deferred behind GPU setup
        self._asked_about_gpu = False
        self._waiting_for_gpu_setup = False
        # Rebound wholesale (never mutated) when the editor saves, so the API's
        # request threads can read it without a lock.
        self._vocab: list[Rule] = load_vocabulary()
        # Loaded once and never rebound, so the API's request threads can read it
        # without a lock either. Plugins cannot be reloaded — see registry.py.
        self._plugins = load_plugins(cfg)
        # Only built where something could actually use it. A plugin's actions are
        # the half that reaches outside PyWhispr, so they have their own switch.
        self._actions = (
            ActionRunner()
            if cfg.plugin_actions_enabled and any(p.act is not None for p in self._plugins)
            else None
        )
        # Filled by the plugin pass, drained once the text has been inserted.
        self._pending_actions: tuple[PendingAction, ...] = ()
        self._fillers = filler_words(cfg.extra_filler_words, cfg.keep_filler_words)
        self._reset_phrases = compile_reset_phrases(cfg.voice_reset_phrases)
        self.injector = TextInjector(cfg.paste_delay_ms, cfg.clipboard_restore_delay_ms)
        self._context = ContextTracker(
            max_chars=cfg.context_chars, memory_seconds=cfg.context_memory_seconds
        )
        self._last_inserted: str | None = None
        # The last few transcripts, so one that auto-pasted into the wrong window
        # can still be recovered. In memory only — see history.py.
        self._history = TranscriptHistory()
        self.recorder = AudioRecorder(device=cfg.input_device, on_level=self._mic_level.emit)
        self.ducker = create_ducker(cfg)
        self.listener = create_hotkey_listener(
            cfg.hotkey,
            on_toggle=self._hotkey_toggled.emit,
            on_release=self._hotkey_released.emit,
        )
        # Its own listener, and one that is never stopped around a dialog the way
        # the dictation one is: pressing it does nothing outside RECORDING, and
        # no dialog can be open while a recording is running.
        self.reset_listener = (
            create_hotkey_listener(cfg.reset_hotkey, on_toggle=self._reset_requested.emit)
            if cfg.reset_hotkey and cfg.reset_hotkey != cfg.hotkey
            else None
        )
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pywhispr-stt")
        self._model_error: str | None = None

        # Network API. Requests run on their own threads but hand the actual
        # transcription to _worker, so the model is still only ever used by one
        # thread at a time and remote work queues behind local dictation.
        # Not in the Lite build: it has no local model to expose — it is itself a
        # client of some other machine's API — so hosting one would only forward.
        self.api = (
            TranscriptionServer(
                transcribe=self._api_transcribe,
                status=self._api_status,
                host=cfg.api_host,
                port=cfg.api_port,
                max_audio_seconds=cfg.api_max_audio_seconds,
                max_queue=cfg.api_max_queue,
            )
            if cfg.api_enabled and not flavor.IS_LITE
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
        self._reset_requested.connect(self._on_reset)
        self._history_requested.connect(self._show_history)
        self._mic_level.connect(self.overlay.on_level)
        self._model_ready.connect(self._on_model_ready)
        self._model_failed.connect(self._on_model_failed)
        self._transcribed.connect(self._on_transcribed)
        self._transcribe_failed.connect(self._on_transcribe_failed)
        self.injector.finished.connect(self._on_insert_finished)

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
                f"{flavor.PRODUCT_NAME}: network API off",
                f"Port {self.cfg.api_port} is already in use. Dictation still works.",
            )
        if not warn_if_missing_permissions():
            self.tray.notify(f"{flavor.PRODUCT_NAME}: clipboard mode", MACOS_PERMISSIONS_HELP)
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
        if self.reset_listener is not None:
            try:
                self.reset_listener.start()
                log.info("Reset hotkey listening on %s", self.cfg.reset_hotkey)
            except Exception as exc:
                # Not even worth a notification: dictation itself still works.
                log.warning("Could not register reset hotkey %r: %s", self.cfg.reset_hotkey, exc)
                self.reset_listener = None

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
        if flavor.IS_LITE:
            # No local model, so none of the download/storage/GPU first-run flows
            # apply — just make sure we know which server to talk to, then connect.
            if not self.cfg.server_url:
                self._prompt_for_server()
            self._begin_model_load()
            return
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
        if not self.cfg.use_gpu:
            # Setting it up is asking for it. The check at the end of the setup runs
            # in a subprocess that reads the config, so leaving the switch off would
            # have it report the CPU and call the install a failure.
            gpu.turn_on(self.cfg)
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
        if flavor.IS_LITE:
            return  # nothing is downloaded: the model lives on the remote server
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
        if self.reset_listener is not None:
            self.reset_listener.stop()
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
        if self._actions is not None:
            self._actions.stop()
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

    # -- remote server (Lite build only) -------------------------------------

    def _prompt_for_server(self) -> None:
        """Ask for the server URL and apply it. Used at first run and from the tray."""
        from pywhispr.ui.server_dialog import ServerDialog

        url = ServerDialog.get_server_url(self.cfg.server_url)
        if url is None or url == self.cfg.server_url:
            return  # cancelled, or unchanged: nothing to rebuild
        self._apply_server_url(url)

    def _set_server(self) -> None:
        """Tray 'Set server…': repoint at a different server and reconnect."""
        self._prompt_for_server()

    def _apply_server_url(self, url: str) -> None:
        """Save the new server, rebuild the backend and (re)connect to it."""
        self.cfg.server_url = url
        save_config(self.cfg)
        log.info("Server set; rebuilding the remote backend")
        self.backend = create_backend(self.cfg)
        self._set_state(State.LOADING)
        self.tray.set_status("Connecting to server…")
        # The load() is only a health check for the remote backend, but running it
        # off the worker keeps the pattern identical to the local backends and off
        # the UI thread.
        self._worker.submit(self._load_model)

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

        Plugins apply too, but only their rewrites — never their actions. This
        port is open to the LAN with no authentication (see ``config.api_host``),
        so anything that can reach it can choose the words that arrive here; a
        plugin's side effects must not be one keyword away from that. Nothing is
        stored on ``self`` either, since this runs on a request thread.
        """
        text = self._worker.submit(self.backend.transcribe, audio).result(
            timeout=QUEUE_TIMEOUT_SECONDS + len(audio) / SAMPLE_RATE
        )
        corrected = self._corrected(self._cleaned(self._after_reset(text)))
        return self._via_plugins(corrected, collect_actions=False)

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
        if not self.cfg.use_gpu:
            return  # switched off on purpose; the tray entry is how it comes back
        providers = session_providers(getattr(self.backend, "_model", None))
        if any(p != "CPUExecutionProvider" for p in providers):
            return  # already accelerated
        if self._acceleration_on_offer() is None:
            return
        self._enable_gpu(asked_by_user=False)

    def _gpu_active(self) -> bool:
        """The tray entry's label: is acceleration installed *and* switched on?"""
        return gpu.active(self.cfg)

    def _enable_gpu(self, asked_by_user: bool = True) -> None:
        from pywhispr import cuda, directml
        from pywhispr.ui.foreground import show_in_front
        from pywhispr.ui.setup_window import ask_to_enable, say_restart_needed

        if self._progress_window is not None and self._progress_window.gpu_running:
            show_in_front(self._progress_window)  # already doing it
            return
        if gpu.installed() and not self.cfg.use_gpu:
            # Switched off from this same entry, libraries still on disk: flipping the
            # flag back is the whole job. Before _acceleration_on_offer(), which says
            # None for an installed path and would otherwise re-run the entire setup.
            gpu.turn_on(self.cfg)
            say_restart_needed("GPU acceleration is switched back on.")
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

    def _disable_gpu(self) -> None:
        """Tray menu: stop using GPU acceleration, keeping the libraries on disk.

        Nothing is reloaded here. onnxruntime resolves a session's providers when it
        is built and cannot be talked out of them afterwards — the same reason
        ``cuda.verify()`` needs a subprocess — so the honest answer is the restart
        notice. ``pywhispr disable-gpu`` is still the one that reclaims the disk.
        """
        from pywhispr import cuda, directml
        from pywhispr.ui.foreground import show_in_front
        from pywhispr.ui.setup_window import ask_to_disable, say_restart_needed

        if self.state not in (State.IDLE, State.LOADING):
            return  # no modal mid-recording, like _change_hotkey
        if self._progress_window is not None and self._progress_window.gpu_running:
            show_in_front(self._progress_window)  # it is being installed right now
            return
        # The dialog takes focus, so what we remembered inserting is no longer
        # behind the caret; and the chord pressed inside it must not start a
        # recording.
        self._context.invalidate()
        self.listener.stop()
        try:
            size = (
                cuda.APPROXIMATE_DOWNLOAD_MB
                if cuda.is_installed()
                else directml.APPROXIMATE_DOWNLOAD_MB
            )
            if not ask_to_disable(download_mb=size):
                return
            gpu.turn_off(self.cfg)
            say_restart_needed("GPU acceleration is switched off; the libraries stay on disk.")
        finally:
            self._resume_listeners()

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
                self._transcribed.emit(text)
            except Exception as exc:
                log.exception("Transcription failed")
                self._transcribe_failed.emit(f"{type(exc).__name__}: {exc}")

        self._worker.submit(transcribe)

    def _on_reset(self) -> None:
        """Reset hotkey: drop the audio so far and carry on recording.

        Ignored outside RECORDING — there is no half-said sentence to throw away
        in any other state, and a failing reset must not end the recording, since
        that would insert the very words the user asked to be rid of.
        """
        if self.state != State.RECORDING:
            log.debug("Reset ignored: %s", self.state.name)
            return
        try:
            self.recorder.reset()
        except Exception:
            log.exception("Could not reset the recording")
            return
        self._max_duration_timer.start()  # the clock starts again with the audio
        self._play(self._start_sound)
        self.overlay.show_status("Starting over…")
        self.tray.set_status("Recording… (started over)", active=True)
        QTimer.singleShot(RESET_FEEDBACK_MS, self._back_to_recording_overlay)

    def _back_to_recording_overlay(self) -> None:
        if self.state == State.RECORDING:
            self.overlay.show_recording()

    def _on_max_duration(self) -> None:
        if self.state == State.RECORDING:
            log.info("Max recording duration reached, stopping")
            self._stop_recording()

    def _on_transcribed(self, text: str) -> None:
        # Before the empty check: a recording of nothing but "um" leaves nothing,
        # and so does one the user talked themselves out of.
        text = self._cleaned(self._after_reset(text))
        if not text.strip():
            log.info("Empty transcription, nothing to insert")
            self._finish_cycle()
            return
        log.info("Transcribed %d characters", len(text))
        self._set_state(State.INSERTING)
        self.overlay.hide_overlay()
        # Vocabulary, then plugins: each can change the opening word, and the join
        # decides about whatever the last of them leaves there.
        corrected = self._via_plugins(self._corrected(text))
        if not corrected.strip():
            # A dictation that was nothing but a command. There is no text to
            # paste and none worth remembering, but the action still runs.
            log.info("Plugins consumed the whole transcript; nothing to insert")
            self._run_pending_actions()
            self._finish_cycle()
            return
        # Remembered before the join, and whatever the insertion does next: the
        # join belongs to the caret this transcript was aimed at, and a paste
        # that went to the wrong window is exactly what the history is for.
        self._history.remember(corrected)
        self._last_inserted = self._joined(corrected)
        self.injector.insert(self._last_inserted)

    def _after_reset(self, text: str) -> str:
        """Honour a spoken "scratch that", degrading to the raw transcript.

        First of the passes, so everything after it works on the words the user
        actually meant to keep. Wrapped like the others — the audio is gone — and
        the tripwire is this pass's own contract: a suffix of what came in.
        """
        if not text or self._reset_phrases is None:
            return text
        try:
            kept = strip_before_reset(text, self._reset_phrases)
        except Exception:
            log.exception("Voice reset failed; using the raw transcript")
            return text
        if not is_suffix_of(text, kept):
            log.error(
                "Voice reset produced %d characters that are not a tail of %d; "
                "using the raw transcript",
                len(kept),
                len(text),
            )
            return text
        return kept

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

    def _via_plugins(self, text: str, *, collect_actions: bool = True) -> str:
        """Let the plugins rewrite the transcript, degrading to it untouched.

        Wrapped like ``_corrected`` and ``_joined``, for the same reason: the audio
        is gone by now. There is no length tripwire here because it would be the
        wrong instrument — "thumbs up emoji" legitimately becomes one character.
        The invariant lives in the engine instead, and is stronger: it splices
        untouched slices of this text around spans it has validated, so text
        outside a claim cannot be disturbed however a plugin misbehaves. What is
        left to catch here is the engine itself failing.

        ``collect_actions=False`` for callers with no dictation cycle to run them
        in — the network API — which also keeps this off ``self`` for a request
        thread.
        """
        if not text or not self._plugins:
            return text
        try:
            result = apply_plugins(text, self._plugins)
        except Exception:
            log.exception("Plugin pass failed; using the transcript as it was")
            return text
        if collect_actions:
            self._pending_actions = result.actions
        return result.text

    def _run_pending_actions(self) -> None:
        """Hand the queued plugin actions to their own thread, the text now placed.

        Never before the insertion: an action that types something, switches window
        or reads the clipboard needs the transcript already where it belongs.
        Drained as it goes, so a recall from the history picker — which comes back
        through the same ``_on_insert_finished`` — cannot fire them a second time.
        """
        pending, self._pending_actions = self._pending_actions, ()
        if not pending:
            return
        if self._actions is None:
            log.debug("Discarding %d plugin action(s): actions are off", len(pending))
            return
        self._actions.dispatch(pending)

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
        # Whether or not it auto-pasted: the user has their text either way, and a
        # plugin's action is about what was said rather than where it landed.
        self._run_pending_actions()
        self._finish_cycle()

    def _finish_cycle(self) -> None:
        self._last_inserted = None
        self._pending_actions = ()  # a cycle that ended another way runs nothing
        self.overlay.hide_overlay()
        self._set_state(State.IDLE)
        self.tray.set_status(f"Ready — press {self.cfg.hotkey} to dictate")

    # -- history recall ------------------------------------------------------

    def _show_history(self) -> None:
        """Offer the last few transcripts and paste the chosen one at the caret.

        The recovery for a dictation that auto-pasted into a window without the
        focus the user thought it had: the audio is gone, but the text is still
        here. The picker itself takes the focus away from the field the text is
        meant for, so the window that had it is remembered and given it back
        before the paste keystroke goes out.
        """
        if self.state != State.IDLE:
            log.debug("Recall ignored: still %s", self.state.name)
            return
        items = list(self._history)
        if not items:
            self.tray.notify("Nothing to recall", "No dictations in this session yet.")
            return

        from pywhispr.ui.foreground import remember_foreground, restore_foreground
        from pywhispr.ui.history_dialog import HistoryDialog

        target = remember_foreground()
        # The picker has the focus, so dictating into it helps nobody.
        self.listener.stop()
        try:
            chosen = HistoryDialog.choose(items)
        finally:
            self._resume_listeners()
        if chosen is None:
            return

        restore_foreground(target)
        # Whatever the caret sits after now is not something we put there.
        self._context.invalidate()
        self._set_state(State.INSERTING)
        self._last_inserted = chosen
        log.info("Re-inserting a remembered transcript (%d characters)", len(chosen))
        QTimer.singleShot(FOCUS_RESTORE_MS, lambda: self.injector.insert(chosen))

    def _resume_listeners(self) -> None:
        """Re-arm the hotkey after a dialog that had to silence it.

        Never raises: losing a dialog's result is annoying, losing the tray app
        is worse.
        """
        try:
            self.listener.start()
        except Exception:
            log.exception("Could not restart the hotkey listener")

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
            self._resume_listeners()

    def _open_plugins_folder(self) -> None:
        """Tray menu: show the plugins folder, creating it and its README if needed.

        No dialog and no reload. A plugin is a file, it is read at startup, and
        saying so plainly beats a manager window that cannot honestly offer to
        apply a change in this process — see registry.py.
        """
        from pywhispr.plugins.registry import ensure_plugins_dir

        try:
            directory = ensure_plugins_dir()
        except OSError as exc:
            log.exception("Could not create the plugins folder")
            self.tray.notify("Plugins folder unavailable", str(exc))
            return
        self.tray.open_path(directory)

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
    app.setApplicationName(flavor.PRODUCT_NAME)
    app.setWindowIcon(QIcon(app_pixmap()))
    app.setQuitOnLastWindowClosed(False)  # tray app: no windows most of the time

    whispr = PyWhisprApp(config)
    whispr.start()
    return app.exec()
