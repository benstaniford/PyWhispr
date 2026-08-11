# PyInstaller runtime hook for the Lite bundle.
#
# Runs before any application import, so pywhispr.flavor reads the flag correctly
# the first time it is imported. This is what makes the bundle *be* PyWhisprLite:
# the source is shared with the full app, and this single environment variable is
# the only thing that differs at runtime. setdefault so a developer can still
# override it when running the frozen binary by hand.
import os

os.environ.setdefault("PYWHISPR_LITE", "1")
