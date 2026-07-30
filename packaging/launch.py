"""Frozen-app entry point (PyInstaller on macOS, cx_Freeze on Windows).

Logging is configured before anything heavy is imported, so that an import
error — a missing DLL, a backend that will not load — is recorded rather than
killing a process nobody can see. See pywhispr.logging_setup for where the
files land.
"""

import sys

from pywhispr.logging_setup import redirect_stdio_if_headless, setup_logging

stdio_path = redirect_stdio_if_headless()
log_file = setup_logging()

import logging  # noqa: E402

from pywhispr.certs import use_system_certificates  # noqa: E402

logging.getLogger(__name__).info(
    "Frozen launch: log=%s, stdio=%s, tls=%s",
    log_file,
    stdio_path or "console",
    use_system_certificates(),
)

from pywhispr.frozen import run  # noqa: E402

sys.exit(run(sys.argv[1:]))
