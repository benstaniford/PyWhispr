import sys

from PySide6.QtCore import Qt

from pywhispr.ui.hotkey_dialog import HotkeyCaptureDialog, key_event_to_chord, pretty_chord

CTRL = Qt.KeyboardModifier.ControlModifier
META = Qt.KeyboardModifier.MetaModifier
ALT = Qt.KeyboardModifier.AltModifier
SHIFT = Qt.KeyboardModifier.ShiftModifier
NONE = Qt.KeyboardModifier.NoModifier


class TestChordConversion:
    def test_darwin_maps_control_modifier_to_cmd(self):
        # Qt swaps ⌘/⌃ on macOS: ControlModifier is the Command key.
        chord = key_event_to_chord(Qt.Key.Key_Space, CTRL | SHIFT, platform="darwin")
        assert chord == "<cmd>+<shift>+<space>"

    def test_windows_maps_control_modifier_to_ctrl(self):
        chord = key_event_to_chord(Qt.Key.Key_Space, CTRL | ALT, platform="win32")
        assert chord == "<ctrl>+<alt>+<space>"

    def test_letters_digits_and_fkeys(self):
        assert key_event_to_chord(Qt.Key.Key_D, META | ALT, platform="darwin").endswith("+d")
        assert key_event_to_chord(Qt.Key.Key_9, ALT, platform="win32") == "<alt>+9"
        assert key_event_to_chord(Qt.Key.Key_F6, CTRL, platform="win32") == "<ctrl>+<f6>"

    def test_bare_modifier_press_gives_none(self):
        assert key_event_to_chord(Qt.Key.Key_Shift, SHIFT, platform="darwin") is None
        assert key_event_to_chord(Qt.Key.Key_Control, CTRL, platform="win32") is None

    def test_unsupported_key_gives_none(self):
        assert key_event_to_chord(Qt.Key.Key_VolumeUp, CTRL, platform="win32") is None

    def test_pretty_chord(self):
        assert pretty_chord("<cmd>+<shift>+<space>") == "Cmd+Shift+Space"
        assert pretty_chord("<ctrl>+<page_down>") == "Ctrl+Page Down"


class TestDialog:
    def test_valid_chord_enables_save(self, qtbot):
        dialog = HotkeyCaptureDialog("<cmd>+<shift>+<space>")
        qtbot.addWidget(dialog)
        modifier = CTRL | SHIFT if sys.platform == "darwin" else CTRL | ALT
        qtbot.keyClick(dialog, Qt.Key.Key_D, modifier)
        assert dialog._chord is not None
        assert dialog._chord.endswith("+d")

    def test_chord_without_modifier_is_rejected(self, qtbot):
        dialog = HotkeyCaptureDialog("<cmd>+<shift>+<space>")
        qtbot.addWidget(dialog)
        qtbot.keyClick(dialog, Qt.Key.Key_D)
        assert dialog._chord is None

    def test_escape_cancels(self, qtbot):
        dialog = HotkeyCaptureDialog("<cmd>+<shift>+<space>")
        qtbot.addWidget(dialog)
        dialog.show()
        qtbot.keyClick(dialog, Qt.Key.Key_Escape)
        assert dialog.result() == dialog.DialogCode.Rejected
