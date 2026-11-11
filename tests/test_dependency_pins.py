"""The pins of the notebook's requirements.yaml, the CI environment and the Colab install cell agree."""

import re
from pathlib import Path

import yaml

import build_notebook

ROOT = Path(__file__).resolve().parents[1]


def _pins(lines):
    pins = {}
    for line in lines:
        match = re.match(r"\s*([A-Za-z0-9_.\-]+)==([^\s;#]+)", line)
        if match:
            pins[match.group(1).lower()] = match.group(2)
    return pins


def test_ci_installs_exactly_the_notebook_pins():
    notebook = yaml.safe_load((ROOT / "notebooks/StarDist_2D/requirements.yaml").read_text())
    app = _pins(notebook["dependencies"])
    ci = _pins((ROOT / "tests/requirements.txt").read_text().splitlines())
    assert {name: ci.get(name) for name in app} == app


def test_colab_installs_the_same_versions():
    notebook = yaml.safe_load((ROOT / "notebooks/StarDist_2D/requirements.yaml").read_text())
    app = _pins(notebook["dependencies"])
    colab = _pins(build_notebook.COLAB_PACKAGES)
    assert {name: app.get(name) for name in colab} == colab


def test_one_python_version_everywhere():
    notebook = yaml.safe_load((ROOT / "notebooks/StarDist_2D/requirements.yaml").read_text())
    assert notebook["python_version"] == "3.12"
    assert "python-version: '3.12'" in (ROOT / ".github/workflows/stardist2d_tests.yml").read_text()
