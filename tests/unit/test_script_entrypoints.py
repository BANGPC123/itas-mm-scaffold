from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "script",
    ["scripts/normalize_regulations.py", "scripts/build_vector_index.py"],
)
def test_direct_script_help_imports_project_modules(script: str):
    result = subprocess.run(
        [sys.executable, "-I", "-X", "utf8", script, "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert "ModuleNotFoundError: No module named 'src'" not in result.stderr
    assert result.returncode == 0, result.stderr
