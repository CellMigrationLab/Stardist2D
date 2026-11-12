"""The version is the same everywhere it is written (MorphoBricks: they move together at a release)."""

import json
import re
from pathlib import Path

import yaml

import build_notebook
import stardist2d

ROOT = Path(__file__).resolve().parents[1]


def test_package_installer_notebook_and_index_agree():
    construct = yaml.safe_load((ROOT / "construct.yaml").read_text())["version"]
    setup = re.search(r'version="([^"]+)"', (ROOT / "setup.py").read_text()).group(1)
    notebook = json.loads((ROOT / "notebooks/StarDist_2D/StarDist_2D_ZeroCostDL4Mic.ipynb").read_text())
    cell = "".join(notebook["cells"][0]["source"])
    current = re.search(r'current_version = "([^"]+)"', cell).group(1)
    index = yaml.safe_load((ROOT / "notebooks/notebook_latest_versions.yaml").read_text())
    versions = {
        "stardist2d.__version__": stardist2d.__version__,
        "construct.yaml": str(construct),
        "setup.py": setup,
        "notebook current_version": current,
        "tools/build_notebook.py": build_notebook.VERSION,
        "notebook_latest_versions.yaml": index["StarDist_2D"]["StarDist_2D_ZeroCostDL4Mic"],
    }
    assert len(set(versions.values())) == 1, versions


def test_the_training_report_and_the_notebook_text_carry_the_version():
    """The training PDF and the notebook's introduction say the release they belong to (audit: they said 1.0.0)."""
    from stardist2d.notebook import training_steps

    assert training_steps.NOTEBOOK_VERSION == stardist2d.__version__
    notebook = (ROOT / "notebooks/StarDist_2D/StarDist_2D_ZeroCostDL4Mic.ipynb").read_text()
    assert f"StarDist2D for LabConstrictor v{stardist2d.__version__}**" in notebook
    stale = set(re.findall(r"StarDist2D for LabConstrictor v([0-9.]+)", notebook)) - {stardist2d.__version__}
    assert not stale, stale

