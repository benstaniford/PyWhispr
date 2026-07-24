import pytest
from PySide6.QtCore import Qt

from pywhispr.ui.overlay import OverlayWindow


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
