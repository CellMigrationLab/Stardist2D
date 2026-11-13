"""Build ``notebooks/StarDist_2D/StarDist_2D_ZeroCostDL4Mic.ipynb`` from the vendored 1.20.8 notebook.

Every markdown cell of 1.20.8 is copied unchanged (Colab-only steps get a "JupyterLab users" note after
them); every code cell becomes a one-line call of the step in :mod:`stardist2d.notebook`, under the
title of the original cell. A version cell (LabConstrictor) comes first. Run after changing a step's
call or a note: ``python tools/build_notebook.py``; ``tests/test_notebook.py`` checks the file is current.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "tests" / "reference" / "StarDist_2D_ZeroCostDL4Mic_1.20.8.ipynb"
NOTEBOOK = ROOT / "notebooks" / "StarDist_2D" / "StarDist_2D_ZeroCostDL4Mic.ipynb"
VERSION = "1.1.2"
NOTEBOOK_NAME = "StarDist_2D_ZeroCostDL4Mic"
REPOSITORY = "CellMigrationLab/Stardist2D"
COLAB_PACKAGES = (
    "tifffile==2024.9.20",
    "csbdeep==0.8.2",
    "stardist==0.9.2",
    "bioimageio.core==0.5.11",
    "importlib-metadata==10.2.0",
    "fpdf2==2.8.8",
    "numcodecs==0.16.6",
    "zarr==3.4.0",
    "tf-keras==2.20.0",
    "tensorflow==2.20.0",
)
"""What 1.20.8 installs on Colab (cell 5), plus importlib-metadata (needed by the BioImage Model Zoo export)."""

STEPS = {
    8: "from stardist2d import notebook\n\nsession = notebook.load_dependencies()",
    11: "notebook.check_gpu()",
    13: "notebook.mount_drive()",
    17: "notebook.download_example_data(session)",
    20: "notebook.training_parameters(session)",
    23: "notebook.data_augmentation(session)",
    25: "notebook.pretrained_weights(session)",
    28: "notebook.prepare_training(session)",
    30: "notebook.start_training(session)",
    32: "notebook.choose_qc_model(session)",
    34: "notebook.loss_curves(session)",
    36: "notebook.quality_control(session)",
    38: "notebook.export_bioimage_model_zoo(session)",
    41: "notebook.predictions(session)",
    43: "notebook.big_data_predictions(session)",
}
"""Original code cell index -> its replacement (the install cell 5 is written by :func:`install_cell`).

Each cell ends with ';' so that Jupyter does not print the returned form ("Form(title=...)") under it (tester
report, 1.1.1)."""

TITLES = {
    20: "Setting main training parameters",
    23: "Data augmentation",
    32: "Choose the model to evaluate",
    38: "Export your model into the BioImage Model Zoo format",
    41: "Generate prediction(s) from unseen dataset",
    43: "Generate prediction(s) from unseen dataset (Big data)",
}
"""Cell titles where 1.20.8's first heading is a field label rather than a title."""

LOCAL_NOTE = "<font size = 4>**JupyterLab users (LabConstrictor app):** {text}"
STARDIST_DOI = "https://doi.org/10.1007/978-3-030-00934-2_30"
"""The published StarDist 2D paper (Schmidt et al., MICCAI 2018, LNCS 11071, pp. 265-273)."""
TEXT_CHANGES = {
    0: (
        (
            "[Schmidt *et al.* in 2018, on arXiv](https://arxiv.org/abs/1806.03535)",
            f"[Schmidt *et al.* in 2018 at MICCAI]({STARDIST_DOI}) (preprint on [arXiv](https://arxiv.org/abs/1806.03535))",
        ),
        (
            "September 2018. (https://arxiv.org/abs/1806.03535)",
            f"September 2018. Lecture Notes in Computer Science 11071, pp. 265-273 ({STARDIST_DOI}; preprint: "
            "https://arxiv.org/abs/1806.03535)",
        ),
    ),
}
"""The only edits to 1.20.8's markdown text, by cell index: the published StarDist paper is cited, not only its arXiv
preprint (owner decision, 1.0.4). Every other markdown cell is kept word for word."""

NOTES = {
    4: "the app installed StarDist and its dependencies already; the cell below does nothing outside Colab.",
    6: "there is no runtime to restart; continue with section 1.3.",
    9: "this section is only needed on Google Colab; the cells below just report what they find.",
    12: "your files are on your computer, so there is no Drive to mount. Use folder paths on your computer below.",
    16: "the app ships a small copy of this dataset (6 training pairs, 2 test pairs), so this works offline. Tick "
    "`download_full_dataset` to download all 45 training pairs from Zenodo. Relative paths below start from "
    "`base_folder` (`~/StarDist2D` by default).",
    27: "if a model of the same name exists, tick `overwrite_existing_model` to replace it (the notebook no longer "
    "deletes it without asking).",
    40: "ImageJ ROI zips are written only when `Region_of_interests` is ticked (ticked by default).",
    44: "your predictions are already on your computer, in `Results_folder`.",
}
EXTRA_LOG = (
    f"<font size = 4>**StarDist2D for LabConstrictor v{VERSION}** (port of v1.20.8): the notebook runs locally in "
    "JupyterLab (LabConstrictor installer) and on Google Colab; the code lives in the `stardist2d` package, also used "
    "by the Fiji, napari and QuPath tools. Fixes and every difference from 1.20.8 are listed in the "
    "[CHANGELOG](https://github.com/CellMigrationLab/Stardist2D/blob/main/CHANGELOG.md)."
)


HIDDEN_CODE = {
    "cellView": "form",
    "collapsed": False,
    "jupyter": {"source_hidden": True},
    "jupyterlabHideCode": {"locked": True},
}
"""Code-cell metadata that the template's hide_code_cells.py sets (CI and the installer run it on every notebook)."""


def _cell(cell_type: str, source: str) -> dict[str, object]:
    """Return a notebook cell (code cells hidden as Colab forms, no outputs)."""
    cell: dict[str, object] = {"cell_type": cell_type, "metadata": {}, "source": source.splitlines(keepends=True)}
    if cell_type == "code":
        cell.update(execution_count=None, outputs=[], metadata=dict(HIDDEN_CODE))
    return cell


def title_of(source: str, default: str) -> str:
    """Return the first ``#@markdown`` heading of an original cell, without the markup."""
    for line in source.splitlines():
        match = re.match(r"\s*#\s*@markdown\s*#*\s*(.+)", line)
        if match:
            return match.group(1).strip()
    return default


def version_cell() -> str:
    """Return the template's version cell (``.tools/docs/notebook_upload.md``) with this notebook's name and version."""
    text = (ROOT / ".tools" / "docs" / "notebook_upload.md").read_text(encoding="utf-8")
    snippet = text.split("```\n", 2)[1]
    snippet = snippet.replace('current_version = "0.0.1"', f'current_version = "{VERSION}"')
    snippet = re.sub(r'notebook_name = "Notebook_template".*', f'notebook_name = "{NOTEBOOK_NAME}"', snippet)
    return "# @title Notebook version\n" + snippet.split("\n", 1)[1].rstrip() + "\n"


def install_cell() -> str:
    """Return section 1.1: install on Colab only (the app installed everything locally), then restart the runtime."""
    packages = ",\n        ".join(f'"{package}"' for package in COLAB_PACKAGES)
    return f"""# @title Install StarDist and dependencies
import subprocess
import sys

if "google.colab" in sys.modules:
    packages = [
        {packages},
        "git+https://github.com/{REPOSITORY}@{VERSION}",
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages], check=True)
    print("Installed: the runtime now restarts (section 1.2).")
    exit(0)
else:
    print("Running locally: StarDist2D and its dependencies were installed with the app.")
"""


def changed_text(index: int, source: str) -> str:
    """Return markdown cell *index* of 1.20.8 with its :data:`TEXT_CHANGES`; refuse an old text not found once.

    Raises
    ------
    ValueError
        If an old text does not occur exactly once (the reference notebook and the change list disagree).
    """
    for old, new in TEXT_CHANGES.get(index, ()):
        if source.count(old) != 1:
            raise ValueError(f"markdown cell {index}: {old!r} occurs {source.count(old)} times, expected once")
        source = source.replace(old, new)
    return source


def build() -> dict[str, object]:
    """Return the notebook as a dictionary."""
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    cells = [_cell("code", version_cell())]
    for index, original in enumerate(reference["cells"]):
        source = "".join(original["source"])
        if original["cell_type"] == "markdown":
            cells.append(_cell("markdown", changed_text(index, source)))
            if index in NOTES:
                cells.append(_cell("markdown", LOCAL_NOTE.format(text=NOTES[index])))
            if index == 45:
                cells.append(_cell("markdown", EXTRA_LOG))
            continue
        code = (
            install_cell()
            if index == 5
            else f"# @title {TITLES.get(index) or title_of(source, 'Run this cell')}\n{STEPS[index]};\n"
        )
        cells.append(_cell("code", code))
    metadata = {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 0}


def rendered() -> str:
    """Return the notebook file's text exactly as nbformat writes it.

    The template's hide_code_cells.py rewrites a notebook with nbformat whenever it changes a cell, so the file is
    generated in that same form, with its metadata already set: the workflow then leaves it unchanged.
    """
    import nbformat

    text = nbformat.writes(nbformat.from_dict(build()))
    return text if text.endswith("\n") else text + "\n"


def main() -> int:
    """Write the notebook; with ``--check``, only report whether the file is current."""
    text = rendered()
    if "--check" in sys.argv:
        current = NOTEBOOK.is_file() and NOTEBOOK.read_text(encoding="utf-8") == text
        print("notebook is current" if current else "notebook is out of date: run python tools/build_notebook.py")
        return 0 if current else 1
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(text, encoding="utf-8")
    print(f"wrote {NOTEBOOK.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
