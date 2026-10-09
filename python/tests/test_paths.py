import importlib
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from vision_inspect import config


@pytest.mark.parametrize("custom_home", [False, True])
def test_data_root_does_not_depend_on_working_directory(tmp_path, custom_home):
    environment = os.environ.copy()
    environment.pop("VISION_INSPECT_HOME", None)
    expected = Path(__file__).resolve().parents[1]
    if custom_home:
        expected = tmp_path / "external-data"
        environment["VISION_INSPECT_HOME"] = str(expected)

    result = subprocess.run(
        [sys.executable, "-c", "from vision_inspect.config import PROJECT_ROOT; print(PROJECT_ROOT)"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )

    assert Path(result.stdout.strip()) == expected


def test_importing_commands_does_not_run_workflows():
    package_dir = Path(config.__file__).parent
    with (
        patch("torch.cuda.is_available", side_effect=AssertionError("GPU accessed during import")),
        patch("torch.load", side_effect=AssertionError("Artifact loaded during import")),
        patch("torch.save", side_effect=AssertionError("Artifact written during import")),
        patch("cv2.imread", side_effect=AssertionError("Image read during import")),
    ):
        for path in (package_dir / "cli").rglob("*.py"):
            if path.name == "__init__.py":
                continue
            module_name = "vision_inspect." + ".".join(path.relative_to(package_dir).with_suffix("").parts)
            module = importlib.import_module(module_name)
            importlib.reload(module)
            assert callable(module.main)
