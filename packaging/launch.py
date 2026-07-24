"""PyInstaller entry point: always start the app (no CLI subcommands)."""

import logging
import sys

from pywhispr.app import run_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

sys.exit(run_app())
