"""The Fiji / napari / QuPath tools (stardist2d_lc_tools), run through the real LabConstrictor-Tools worker."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.slow


def _tools(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "TF_CPP_MIN_LOG_LEVEL": "3"}
    command = [sys.executable, "-m", "labconstrictor_tools", *arguments, "--module", "stardist2d_lc_tools"]
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True, env=environment, timeout=1500)


def test_declarations_and_hosts_are_valid():
    result = _tools("check", "--pythonpath", "src", "--app-manifest", "labconstrictor-app.yaml")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "5 tool(s)" in result.stdout
    assert "hosts (labconstrictor-app.yaml): notebook, napari, fiji, qupath" in result.stdout


def test_every_tool_case_passes():
    result = _tools("test", "--pythonpath", "src", "--cases", "lc_tests/cases.json")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "0 failed" in result.stdout and "with no test case" not in result.stdout


def test_cancel_stops_a_long_segmentation(tmp_path):
    """A tool that ignores Cancel is killed by the hosts after 3 s; this one stops between tiles."""
    nuclei = tifffile.imread(ROOT / "lc_tests" / "fixtures" / "nuclei.tif")
    tifffile.imwrite(tmp_path / "big.tif", np.tile(nuclei, (6, 6)))
    cases = [{"tool": "segment_fluorescence", "inputs": {"image": "big.tif"}, "cancel_after_s": 2.0}]
    (tmp_path / "cases.json").write_text(json.dumps(cases))
    result = _tools(
        "test", "--pythonpath", str(ROOT / "src"), "--cases", str(tmp_path / "cases.json"), "--check-cancel"
    )
    assert result.returncode == 0, result.stdout + result.stderr
