from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtCore import Qt

from pywhispr.ui.overlay import OverlayWindow

HWND_TOP = 0
HWND_TOPMOST = -1
SWP_NOACTIVATE = 0x0010


@pytest.fixture
def user32():
    """Fake user32 so the z-order calls can be asserted off Windows too."""
    fake = MagicMock()
    ctypes = MagicMock()
    ctypes.windll.user32 = fake
    with patch.dict("sys.modules", {"ctypes": ctypes}), patch(
        "pywhispr.ui.overlay.QGuiApplication.platformName", return_value="windows"
    ):
        yield fake


def insert_after_args(user32):
    return [call.args[1] for call in user32.SetWindowPos.call_args_list]


@pytest.fixture
def overlay(qtbot):
    w = OverlayWindow()
    qtbot.addWidget(w)
    return w


def test_overlay_never_takes_focus(overlay):
    flags = overlay.windowFlags()
    assert flags & Qt.WindowType.FramelessWindowHint
    assert flags & Qt.WindowType.WindowStaysOnTopHint
    assert flags & Qt.WindowType.WindowDoesNotAcceptFocus
    assert flags & Qt.WindowType.WindowTransparentForInput
    assert overlay.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    assert overlay.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)


def test_recording_then_status_then_hide(overlay):
    overlay.show_recording()
    assert overlay.waveform.isVisibleTo(overlay)
    assert not overlay.status_label.isVisibleTo(overlay)

    overlay.on_level(0.5)  # must not raise

    overlay.show_status("Transcribing…")
    assert overlay.status_label.isVisibleTo(overlay)
    assert not overlay.waveform.isVisibleTo(overlay)

    overlay.hide_overlay()
    assert overlay.isHidden()


def test_show_claims_the_top_of_the_topmost_band(overlay, user32):
    """Topmost alone loses the tie against another topmost window (a docked
    Deckmaster), and HWND_TOPMOST is a no-op once we are already topmost — so
    the ordering claim has to be HWND_TOP. HWND_TOPMOST is still sent first, to
    re-establish the ex-style after a hide/show."""
    overlay.show_recording()
    assert insert_after_args(user32) == [HWND_TOPMOST, HWND_TOP]

    # Something else claims the top mid-recording; the timer takes it back.
    assert overlay._topmost_timer.isActive()
    user32.reset_mock()
    overlay._keep_on_top()
    assert HWND_TOP in insert_after_args(user32)

    # ...and a hide/show cycle claims it again.
    overlay.hide_overlay()
    assert not overlay._topmost_timer.isActive()
    user32.reset_mock()
    overlay.show_recording()
    assert insert_after_args(user32) == [HWND_TOPMOST, HWND_TOP]


def test_raising_never_activates(overlay, user32):
    overlay.show_recording()
    for call in user32.SetWindowPos.call_args_list:
        assert call.args[6] & SWP_NOACTIVATE
    user32.SetForegroundWindow.assert_not_called()
    assert not overlay.isActiveWindow()


def test_z_order_failure_does_not_break_the_overlay(overlay, user32):
    user32.SetWindowPos.side_effect = OSError("boom")
    overlay.show_recording()  # must not raise
    assert overlay.isVisible()
