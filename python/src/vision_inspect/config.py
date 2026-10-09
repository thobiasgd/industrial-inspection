import os
from pathlib import Path


PROJECT_ROOT = Path(
    os.environ.get("VISION_INSPECT_HOME", Path(__file__).resolve().parents[2])
).expanduser().resolve()
