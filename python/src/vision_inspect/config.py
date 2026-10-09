"""Resolve local data independently of the terminal's working directory."""

import os
from pathlib import Path


# The default is python/ for an editable installation of this repository.
# For a wheel installation, point VISION_INSPECT_HOME to the data directory.
PROJECT_ROOT = Path(
    os.environ.get("VISION_INSPECT_HOME", Path(__file__).resolve().parents[2])
).expanduser().resolve()
