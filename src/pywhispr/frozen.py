"""What the packaged executable does with its arguments.

Lives here rather than in ``packaging/launch.py`` so it can be tested: the frozen
entry point runs at import time, by design, so that an import failure is logged.
"""

from __future__ import annotations


def run(arguments: list[str]) -> int:
    """Dispatch the frozen executable: a subcommand, or the app itself.

    The exe *is* the command in a packaged install — ``cuda.verify()`` re-runs
    ``sys.executable verify-gpu``. Ignoring the arguments started a second copy of
    the whole app: another tray icon, another model download, another progress
    window beside the first.
    """
    if arguments:
        from pywhispr.cli import main

        return main(arguments)

    from pywhispr.app import run_app

    return run_app()


__all__ = ["run"]
