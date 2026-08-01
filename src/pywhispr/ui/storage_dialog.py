"""Asked once, before the first download: where do the gigabytes go?

The model and the CUDA libraries are about 6 GB together and they default to the
user profile on the system drive. A config key covers the case where someone
knows to set it, but on a fresh install nothing has been read from the config by
the time the download starts, so a machine with a small C: fills up before the
user has a chance. Hence a question, and only when it might matter: the default
target has to be short of room.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QMessageBox

from pywhispr.storage import REQUIRED_MB, default_model_dir, free_mb
from pywhispr.ui.foreground import show_in_front

log = logging.getLogger(__name__)


def should_ask(required_mb: int = REQUIRED_MB) -> bool:
    """Only when the default location cannot comfortably hold the downloads.

    Unknown free space counts as fine: this is a convenience, and a question
    nobody needed is worse than a check that stayed quiet.
    """
    free = free_mb(default_model_dir())
    return free is not None and free < required_mb


def wants_to_change(parent=None, required_mb: int = REQUIRED_MB) -> bool:
    """Does the user want the downloads somewhere other than the default?"""
    default = default_model_dir()
    free = free_mb(default)
    box = QMessageBox(parent)
    box.setWindowTitle("PyWhispr — where to keep the downloads")
    box.setIcon(QMessageBox.Icon.Warning)
    box.setText("PyWhispr is about to download the speech model.")
    box.setInformativeText(
        f"It needs roughly {required_mb // 1000} GB with GPU support, and "
        f"{default} has {'unknown' if free is None else str(free // 1000) + ' GB'} free.\n\n"
        "Pick another drive, or carry on here."
    )
    choose = box.addButton("Choose a folder…", QMessageBox.ButtonRole.AcceptRole)
    box.addButton("Use the default", QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(choose)
    show_in_front(box)
    box.exec()
    return box.clickedButton() is choose


def browse_for_directory(parent=None) -> str:
    """The folder picker, or "" if it was cancelled."""
    return QFileDialog.getExistingDirectory(
        parent,
        "Where should PyWhispr keep the model and CUDA libraries?",
        str(Path.home()),
    )


def ask_where_to_store(parent=None, required_mb: int = REQUIRED_MB) -> Path | None:
    """Offer to put the downloads elsewhere. Returns the chosen base directory.

    None means "leave the defaults alone", which is also what cancelling the
    browser means — the download still works, it just goes where it always did.

    The two dialogs are separate functions rather than inline, so a test can answer
    them without patching Qt's own methods: replacing ``QMessageBox.exec`` on the
    class killed the interpreter outright.
    """
    if not wants_to_change(parent, required_mb):
        return None

    chosen = browse_for_directory(parent)
    if not chosen:
        return None

    target = Path(chosen) / "PyWhispr"
    room = free_mb(target)
    if room is not None and room < required_mb:
        # Said, not enforced: an external drive may be the only option they have,
        # and the estimate includes GPU libraries they might decline.
        QMessageBox.warning(
            parent,
            "PyWhispr — still tight",
            f"{target} has about {room // 1000} GB free, "
            f"less than the {required_mb // 1000} GB this may need. Using it anyway.",
        )
    log.info("Storing downloads under a chosen directory instead of the default")
    return target


__all__ = ["ask_where_to_store", "browse_for_directory", "should_ask", "wants_to_change"]
