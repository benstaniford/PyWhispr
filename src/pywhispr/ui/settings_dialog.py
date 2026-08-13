"""The settings window: one place for everything the tray menu used to carry.

The tray keeps what a tray is for — start/stop, recall, open this window, quit —
and every preference lives here instead. Deliberately not every config key: the
model, thread and paste-timing knobs stay in config.toml, which "Open config
file" on the Advanced tab reaches, because they are typed once in a lifetime and
a spin box each would bury the settings people do change.

The dialog only edits a *copy* of the config and hands it back on Save; deciding
what can be applied without a restart is the app's job (see app._apply_settings).
"""

from __future__ import annotations

import dataclasses
import logging
import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from pywhispr import flavor
from pywhispr.audio import all_input_devices, display_name, input_devices
from pywhispr.config import Config

log = logging.getLogger(__name__)

# Exactly the fields this window edits. The app copies these across rather than
# swapping the whole config, because the GPU buttons on the Advanced tab change
# the *live* config while the window is open (gpu.turn_off writes use_gpu and
# saves) — a wholesale replace would put this window's older copy back over it.
EDITED_FIELDS = (
    "hotkey",
    "reset_hotkey",
    "input_device",
    "input_device_name",
    "max_recording_seconds",
    "play_sounds",
    "duck_other_audio",
    "remove_fillers",
    "join_continuations",
    "lowercase_continuations",
    "vocabulary_enabled",
    "vocabulary_fuzzy",
    "voice_reset_phrases",
    "plugins_enabled",
    "plugin_actions_enabled",
    "api_enabled",
    "api_host",
    "api_port",
    "server_url",
)

SYSTEM_DEFAULT = "System default"
# A configured microphone that is not plugged in right now still has to be
# offered, or opening the settings would quietly reset it to the default.
MISSING_SUFFIX = " (not connected)"


class SettingsDialog(QDialog):
    """Modal editor over a copy of the config.

    ``actions`` are the things that are not a value: the hotkey capture, the
    vocabulary editor, the plugins folder, the GPU setup, the config and log
    files. Each is optional — the Lite build has no GPU path and the full app has
    no server — and a missing one leaves its row out.
    """

    def __init__(
        self,
        cfg: Config,
        *,
        on_change_hotkey=None,
        on_change_reset_hotkey=None,
        on_edit_vocabulary=None,
        on_open_plugins=None,
        on_enable_gpu=None,
        on_disable_gpu=None,
        gpu_active=None,
        on_open_config=None,
        on_open_log=None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(f"{flavor.PRODUCT_NAME} Settings")
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.setMinimumWidth(460)
        # A copy: cancelling has to leave the running app's config untouched.
        self.config = dataclasses.replace(cfg)
        self._on_change_hotkey = on_change_hotkey
        self._on_change_reset_hotkey = on_change_reset_hotkey
        self._on_enable_gpu = on_enable_gpu
        self._on_disable_gpu = on_disable_gpu
        self._gpu_active = gpu_active

        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        tabs.addTab(self._dictation_tab(), "Dictation")
        tabs.addTab(self._text_tab(on_edit_vocabulary), "Text")
        tabs.addTab(self._advanced_tab(on_open_plugins, on_open_config, on_open_log), "Advanced")
        layout.addWidget(tabs)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # -- tabs ----------------------------------------------------------------

    def _dictation_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)

        self._hotkey_label = QLabel(self.config.hotkey)
        form.addRow("Hotkey", self._with_button(self._hotkey_label, "Change…", self._change_hotkey))

        self._reset_hotkey_label = QLabel(self.config.reset_hotkey or "off")
        form.addRow(
            "Start-over hotkey",
            self._with_button(self._reset_hotkey_label, "Change…", self._change_reset_hotkey),
        )

        self._mic = QComboBox()
        self._mic_note = QLabel()
        self._mic_note.setStyleSheet("color: gray;")
        self._mic_note.setWordWrap(True)
        self._fill_microphones()
        form.addRow("Microphone", self._with_button(self._mic, "Refresh", self._fill_microphones))
        form.addRow("", self._mic_note)

        self._max_seconds = QSpinBox()
        self._max_seconds.setRange(5, 3600)
        self._max_seconds.setSuffix(" s")
        self._max_seconds.setValue(self.config.max_recording_seconds)
        form.addRow("Stop recording after", self._max_seconds)

        self._play_sounds = self._check("Play a sound when recording starts and stops", "play_sounds")
        form.addRow("", self._play_sounds)

        self._duck = self._check("Quieten other applications while recording", "duck_other_audio")
        self._duck.setEnabled(sys.platform == "win32")
        if sys.platform != "win32":
            self._duck.setToolTip("Windows only")
        form.addRow("", self._duck)
        return page

    def _text_tab(self, on_edit_vocabulary) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)

        self._remove_fillers = self._check('Remove hesitations ("um", "uh")', "remove_fillers")
        form.addRow("Cleanup", self._remove_fillers)

        self._join = self._check("Join consecutive dictations into one passage", "join_continuations")
        form.addRow("", self._join)
        self._lowercase = self._check(
            "Lower-case a continuing word", "lowercase_continuations"
        )
        self._join.toggled.connect(self._lowercase.setEnabled)
        self._lowercase.setEnabled(self._join.isChecked())
        form.addRow("", self._lowercase)

        self._vocab_enabled = self._check("Correct spellings from my vocabulary", "vocabulary_enabled")
        form.addRow("Vocabulary", self._vocab_enabled)
        self._vocab_fuzzy = self._check("Also fix near misses on longer terms", "vocabulary_fuzzy")
        self._vocab_enabled.toggled.connect(self._vocab_fuzzy.setEnabled)
        self._vocab_fuzzy.setEnabled(self._vocab_enabled.isChecked())
        if on_edit_vocabulary is not None:
            form.addRow("", self._with_button(self._vocab_fuzzy, "Edit vocabulary…", on_edit_vocabulary))
        else:
            form.addRow("", self._vocab_fuzzy)

        self._reset_phrases = QLineEdit(", ".join(self.config.voice_reset_phrases))
        self._reset_phrases.setPlaceholderText("clear clear")
        form.addRow("Start-over phrases", self._reset_phrases)
        hint = QLabel(
            "Say one of these and only what follows it is kept. Comma-separated; "
            "empty turns it off."
        )
        hint.setStyleSheet("color: gray;")
        hint.setWordWrap(True)
        form.addRow("", hint)
        return page

    def _advanced_tab(self, on_open_plugins, on_open_config, on_open_log) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)

        self._plugins_enabled = self._check("Run plugins", "plugins_enabled")
        form.addRow("Plugins", self._plugins_enabled)
        self._plugin_actions = self._check(
            "Let plugins do things as well as rewrite text", "plugin_actions_enabled"
        )
        self._plugins_enabled.toggled.connect(self._plugin_actions.setEnabled)
        self._plugin_actions.setEnabled(self._plugins_enabled.isChecked())
        if on_open_plugins is not None:
            form.addRow("", self._with_button(self._plugin_actions, "Open folder…", on_open_plugins))
        else:
            form.addRow("", self._plugin_actions)

        if self._on_enable_gpu is not None:
            self._gpu_button = QPushButton()
            self._gpu_button.clicked.connect(self._gpu_clicked)
            self._refresh_gpu_button()
            form.addRow("GPU acceleration", self._gpu_button)

        if flavor.IS_LITE:
            self._server_url = QLineEdit(self.config.server_url)
            form.addRow("Transcription server", self._server_url)
        else:
            self._api_enabled = self._check("Accept transcription requests over the network", "api_enabled")
            form.addRow("Network API", self._api_enabled)
            self._api_host = QLineEdit(self.config.api_host)
            self._api_host.setToolTip("0.0.0.0 accepts from the whole network; 127.0.0.1 is this machine only")
            form.addRow("Listen on", self._api_host)
            self._api_port = QSpinBox()
            self._api_port.setRange(1, 65535)
            self._api_port.setValue(self.config.api_port)
            form.addRow("Port", self._api_port)

        files = QHBoxLayout()
        if on_open_config is not None:
            config_button = QPushButton("Open config file")
            config_button.clicked.connect(lambda: on_open_config())
            files.addWidget(config_button)
        if on_open_log is not None:
            log_button = QPushButton("Open log file")
            log_button.clicked.connect(lambda: on_open_log())
            files.addWidget(log_button)
        files.addStretch()
        holder = QWidget()
        holder.setLayout(files)
        form.addRow("Files", holder)
        return page

    # -- helpers -------------------------------------------------------------

    def _check(self, text: str, field: str) -> QCheckBox:
        box = QCheckBox(text)
        box.setChecked(bool(getattr(self.config, field)))
        return box

    @staticmethod
    def _with_button(widget, text: str, handler) -> QWidget:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(widget, 1)
        button = QPushButton(text)
        button.clicked.connect(lambda: handler())  # swallow triggered(bool)
        row.addWidget(button)
        holder = QWidget()
        holder.setLayout(row)
        return holder

    def _fill_microphones(self) -> None:
        """(Re)list the input devices, keeping whatever is currently chosen.

        A device named in the config but absent right now is listed anyway, marked,
        so that opening this window while it is unplugged does not silently reset
        the choice to the default.
        """
        chosen = self._chosen_device_name()
        self._mic.clear()
        self._mic.addItem(SYSTEM_DEFAULT, None)
        names = [name for _index, name in input_devices()]
        for name in names:
            self._mic.addItem(name, name)
        if chosen and chosen not in names:
            self._mic.addItem(chosen + MISSING_SUFFIX, chosen)
        index = self._mic.findData(chosen)
        self._mic.setCurrentIndex(index if index >= 0 else 0)
        if chosen and chosen not in names:
            self._mic_note.setText(
                f"{chosen} is not connected. Recording uses the system default until it is back."
            )
        elif not names:
            self._mic_note.setText("No input devices found.")
        else:
            self._mic_note.setText("")

    def _chosen_device_name(self) -> str | None:
        """The configured microphone by name, resolving a legacy index if that is
        all the config has."""
        if self.config.input_device_name:
            return display_name(self.config.input_device_name)
        if self.config.input_device is None:
            return None
        for index, name in all_input_devices():
            if index == self.config.input_device:
                return display_name(name)
        return None

    def _change_hotkey(self) -> None:
        if self._on_change_hotkey is None:
            return
        chord = self._on_change_hotkey(self.config.hotkey)
        if chord:
            self.config.hotkey = chord
            self._hotkey_label.setText(chord)

    def _change_reset_hotkey(self) -> None:
        if self._on_change_reset_hotkey is None:
            return
        chord = self._on_change_reset_hotkey(self.config.reset_hotkey)
        if chord:
            self.config.reset_hotkey = chord
            self._reset_hotkey_label.setText(chord)

    def _gpu_is_active(self) -> bool:
        """Anything unanswerable counts as off — same reasoning as the tray's old
        entry: a settings page that will not open is worse than a declined offer."""
        if self._gpu_active is None:
            return False
        try:
            return bool(self._gpu_active())
        except Exception:
            log.debug("Could not tell whether GPU acceleration is on", exc_info=True)
            return False

    def _refresh_gpu_button(self) -> None:
        self._gpu_button.setText("Disable…" if self._gpu_is_active() else "Enable…")

    def _gpu_clicked(self) -> None:
        """Asked again now rather than read off the label."""
        handler = self._on_disable_gpu if self._gpu_is_active() else self._on_enable_gpu
        if handler is not None:
            handler()
        self._refresh_gpu_button()

    # -- saving --------------------------------------------------------------

    def _save(self) -> None:
        cfg = self.config
        cfg.input_device_name = self._mic.currentData()
        # The legacy index would otherwise outlive the choice made here for anyone
        # who picks "System default" after having had one set.
        cfg.input_device = None
        cfg.max_recording_seconds = self._max_seconds.value()
        cfg.play_sounds = self._play_sounds.isChecked()
        cfg.duck_other_audio = self._duck.isChecked()
        cfg.remove_fillers = self._remove_fillers.isChecked()
        cfg.join_continuations = self._join.isChecked()
        cfg.lowercase_continuations = self._lowercase.isChecked()
        cfg.vocabulary_enabled = self._vocab_enabled.isChecked()
        cfg.vocabulary_fuzzy = self._vocab_fuzzy.isChecked()
        cfg.voice_reset_phrases = [
            phrase.strip() for phrase in self._reset_phrases.text().split(",") if phrase.strip()
        ]
        cfg.plugins_enabled = self._plugins_enabled.isChecked()
        cfg.plugin_actions_enabled = self._plugin_actions.isChecked()
        if flavor.IS_LITE:
            cfg.server_url = self._server_url.text().strip()
        else:
            cfg.api_enabled = self._api_enabled.isChecked()
            cfg.api_host = self._api_host.text().strip() or "0.0.0.0"
            cfg.api_port = self._api_port.value()
        self.accept()

    @staticmethod
    def edit(cfg: Config, **actions) -> Config | None:
        """Show the dialog; return the edited config, or None if it was cancelled."""
        dialog = SettingsDialog(cfg, **actions)
        dialog.raise_()
        dialog.activateWindow()
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.config
        return None
