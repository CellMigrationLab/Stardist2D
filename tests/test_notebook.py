"""The notebook: built from 1.20.8 (all text kept), portable, and every original parameter in a form."""

import ast
import json
import re
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

import build_notebook
from reference_notebook import cells as reference_cells
from stardist2d import notebook

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "StarDist_2D" / "StarDist_2D_ZeroCostDL4Mic.ipynb"


def _cells():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def test_the_notebook_file_is_current():
    assert subprocess.run([sys.executable, str(ROOT / "tools" / "build_notebook.py"), "--check"]).returncode == 0


def test_every_markdown_cell_of_1_20_8_is_kept_unchanged_and_in_order():
    """Word for word, except the documented TEXT_CHANGES (the published StarDist paper is cited)."""
    original = [
        build_notebook.changed_text(index, "".join(c["source"]))
        for index, c in enumerate(reference_cells())
        if c["cell_type"] == "markdown"
    ]
    ours = ["".join(c["source"]) for c in _cells() if c["cell_type"] == "markdown"]
    iterator = iter(ours)
    assert all(any(text == candidate for candidate in iterator) for text in original)


def test_the_published_stardist_paper_is_cited():
    intro = "".join(_cells()[1]["source"])
    assert intro.count(build_notebook.STARDIST_DOI) == 2
    assert "on arXiv](" not in intro and "preprint on [arXiv](https://arxiv.org/abs/1806.03535)" in intro
    changes = [change for changes in build_notebook.TEXT_CHANGES.values() for change in changes]
    assert len(changes) == 2  # any new edit of 1.20.8's text must be added here and in the CHANGELOG


def test_code_cells_are_titled_short_and_portable():
    for cell in _cells():
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        assert source.startswith("# @title "), source[:40]
        assert not re.search(r"^\s*[!%]", source, re.MULTILINE), "no IPython magics"
        assert cell["outputs"] == [] and cell["execution_count"] is None
        ast.parse(source)
        if "google.colab" in source and "notebook_name" not in source:
            assert 'if "google.colab" in sys.modules:' in source


def test_the_version_cell_names_the_notebook_and_version():
    first = "".join(_cells()[0]["source"])
    assert f'current_version = "{build_notebook.VERSION}"' in first
    assert 'notebook_name = "StarDist_2D_ZeroCostDL4Mic"' in first
    assert NOTEBOOK.stem == "StarDist_2D_ZeroCostDL4Mic"


def _original_parameters():
    names = set()
    for cell in reference_cells():
        for line in "".join(cell["source"]).splitlines():
            match = re.match(r"\s*([A-Za-z_]\w*)\s*=.*#\s*@param", line)
            if match:
                names.add(match.group(1))
    return names


def _all_forms():
    session = notebook.Session(base_folder=Path("/tmp"))
    with mock.patch("stardist2d.notebook.forms.display"):
        return [
            step(session)
            for step in (
                notebook.download_example_data,
                notebook.training_parameters,
                notebook.data_augmentation,
                notebook.pretrained_weights,
                notebook.prepare_training,
                notebook.start_training,
                notebook.choose_qc_model,
                notebook.loss_curves,
                notebook.quality_control,
                notebook.export_bioimage_model_zoo,
                notebook.predictions,
                notebook.big_data_predictions,
            )
        ]


def test_every_original_form_parameter_is_in_a_form():
    fields = {name for form in _all_forms() for name in form.controls}
    assert _original_parameters() - fields == set()


def test_original_defaults_are_kept():
    defaults = {name: control.value for form in _all_forms() for name, control in form.controls.items()}
    assert defaults["number_of_epochs"] == 100 and defaults["patch_size"] == 512 and defaults["grid_parameter"] == 2
    assert (
        defaults["Multiply_dataset_by"] == 4
        and defaults["pretrained_model_choice"] == "2D_versatile_fluo_from_Stardist_Fiji"
    )
    assert (
        defaults["block_size_Y"] == 512 and defaults["min_overlap"] == 50 and defaults["Data_type"] == "Single_Images"
    )


def test_an_expected_problem_is_one_line_not_a_traceback():
    with mock.patch("stardist2d.notebook.forms.display"):
        form = notebook.training_parameters(notebook.Session(base_folder=Path("/tmp")))
    assert form.set(model_name="").run() is False


@pytest.mark.slow
def test_the_notebook_executes_in_a_kernel(tmp_path, monkeypatch):
    """Every cell runs (the forms appear; nothing heavy runs before a button is clicked)."""
    import nbformat

    monkeypatch.setenv("PYTHONPATH", str(ROOT / "src"))
    from nbclient import NotebookClient

    book = nbformat.read(NOTEBOOK, as_version=4)
    NotebookClient(book, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(tmp_path)}}).execute()
    errors = [o for c in book.cells if c.cell_type == "code" for o in c.outputs if o.output_type == "error"]
    assert errors == []


def test_step_cells_end_with_a_semicolon():
    """Jupyter would print the returned form ("Form(title=...)") under a cell without it (tester report, 1.1.1)."""
    steps = [cell for cell in _cells() if cell["cell_type"] == "code" and "notebook." in "".join(cell["source"])]
    assert steps and all("".join(cell["source"]).rstrip().endswith(";") for cell in steps)


def test_a_running_form_shows_it_and_cannot_be_started_twice():
    from stardist2d.notebook.forms import Field, Form

    seen = []
    def record(values):
        seen.append((form.button.disabled, form.button.description))

    form = Form("Step", [Field("x", 1, "int")], record)
    assert form.run() and seen == [(True, "Running…")]
    assert not form.button.disabled and form.button.description == "Run"


def test_a_failing_form_gives_its_button_back():
    from stardist2d.notebook.forms import Form

    def fail(values):
        raise ValueError("bad folder")

    form = Form("Step", [], fail)
    assert not form.run() and not form.button.disabled and form.button.description == "Run"

