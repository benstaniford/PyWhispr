import sys
from unittest.mock import MagicMock, patch

import pytest

from pywhispr import priority


@pytest.fixture
def kernel32():
    """A stand-in for kernel32, so this runs on any platform."""
    fake = MagicMock()
    fake.GetCurrentProcess.return_value = 7
    fake.GetPriorityClass.return_value = 0x20  # NORMAL_PRIORITY_CLASS
    fake.SetPriorityClass.return_value = 1
    with patch.object(sys, "platform", "win32"), patch("ctypes.windll", create=True) as windll:
        windll.kernel32 = fake
        yield fake


def test_raises_then_restores_what_was_there_before(kernel32):
    with priority.boosted():
        assert kernel32.SetPriorityClass.call_args.args == (
            7,
            priority.ABOVE_NORMAL_PRIORITY_CLASS,
        )
    assert kernel32.SetPriorityClass.call_args.args == (7, 0x20)


def test_restores_even_when_the_block_raises(kernel32):
    with pytest.raises(RuntimeError), priority.boosted():
        raise RuntimeError("transcription blew up")
    assert kernel32.SetPriorityClass.call_args.args == (7, 0x20)


def test_a_refused_raise_is_not_undone_and_does_not_stop_the_work(kernel32):
    """A scheduling nicety must never cost a transcript."""
    kernel32.SetPriorityClass.return_value = 0
    ran = False
    with priority.boosted():
        ran = True
    assert ran
    assert kernel32.SetPriorityClass.call_count == 1  # no restore to guess at


def test_non_windows_is_a_no_op():
    with patch.object(sys, "platform", "darwin"), priority.boosted():
        pass
