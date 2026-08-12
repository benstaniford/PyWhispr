"""Running the side-effect half of a plugin, off the GUI thread and after the paste.

Three properties, each of which is a bug that would otherwise be waiting to
happen:

- **Its own thread, not the STT worker.** A plugin that spends ten seconds on an
  HTTP call must not put the next dictation ten seconds behind it.
- **One at a time.** Actions run in the order their triggers appeared. A pool
  would let two plugins fight over the keyboard or the clipboard.
- **After the insertion, never before.** An action that types something, switches
  window or reads the clipboard has to happen once the transcript is already in
  place — see ``app._on_insert_finished``.

Exceptions are logged and dropped. An action is by definition the part of a
dictation that has already produced its text, so a failing one must not touch the
state machine.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

from pywhispr.plugins.engine import PendingAction

log = logging.getLogger(__name__)


class ActionRunner:
    """A single background thread that runs plugin actions one after another."""

    def __init__(self) -> None:
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pywhispr-plugin")

    def dispatch(self, actions: Iterable[PendingAction]) -> int:
        """Queue every action. Returns how many were queued; never raises."""
        queued = 0
        for pending in actions:
            if pending.plugin.act is None:
                continue
            try:
                self._pool.submit(self._run, pending)
            except RuntimeError:
                # Submitted after shutdown, i.e. the app is on its way out.
                log.debug("Not running plugin %r: shutting down", pending.plugin.name)
                continue
            queued += 1
        if queued:
            log.debug("Queued %d plugin action(s)", queued)
        return queued

    @staticmethod
    def _run(pending: PendingAction) -> None:
        name = pending.plugin.name
        log.info("Running plugin action %r", name)
        try:
            pending.plugin.act(pending.match)
        except Exception:
            # Never quotes the match: it carries the whole transcript.
            log.exception("Plugin action %r failed", name)
        else:
            log.debug("Plugin action %r finished", name)

    def stop(self) -> None:
        """Drop anything queued and stop. Quitting must not hang on a plugin."""
        self._pool.shutdown(wait=False, cancel_futures=True)
