"""The notebook's steps, driven as a person would (fill the form, click the button), on the shipped example data."""

import contextlib
from unittest import mock

import pytest

from stardist2d import notebook
from stardist2d.notebook import evaluation_steps, forms, prediction_steps, setup_steps, training_steps

pytestmark = pytest.mark.slow


def test_the_whole_notebook_workflow_on_the_example_dataset(tmp_path):
    with contextlib.ExitStack() as stack:
        for module in (forms, setup_steps, training_steps, evaluation_steps, prediction_steps):
            for name in ("display", "show_figure", "show_table"):
                if hasattr(module, name):
                    stack.enter_context(mock.patch.object(module, name))
        session = notebook.Session(base_folder=tmp_path)
        steps = [
            (notebook.download_example_data, {"base_folder": str(tmp_path)}),
            (
                notebook.training_parameters,
                {
                    "model_name": "demo",
                    "number_of_epochs": 1,
                    "Use_Default_Advanced_Parameters": False,
                    "number_of_steps": 2,
                    "patch_size": 256,
                    "batch_size": 2,
                },
            ),
            (notebook.data_augmentation, {}),
            (notebook.pretrained_weights, {}),
            (notebook.prepare_training, {}),
            (notebook.start_training, {}),
            (notebook.choose_qc_model, {}),
            (notebook.loss_curves, {}),
            (notebook.quality_control, {}),
            (
                notebook.export_bioimage_model_zoo,
                {"Trained_model_name": "demo model", "Trained_model_authors": "[Ada]"},
            ),
            (notebook.predictions, {"Data_folder": "data/Stardist/Test - Images", "Results_folder": "results"}),
        ]
        for step, values in steps:
            form = step(session).set(**values)
            assert form.run(), f"{step.__name__}: {form.output.outputs}"

    model = tmp_path / "models" / "demo"
    expected = [
        model / "config.json",
        model / "weights_best.h5",
        model / "thresholds.json",
        model / "demo_training_report.pdf",
        model / "Quality Control" / "training_evaluation.csv",
        model / "Quality Control" / "lossCurvePlots.png",
        model / "Quality Control" / "Quality_Control for demo.csv",
        model / "Quality Control" / "QC_example_data.png",
        model / "Quality Control" / "demo_QC_report.pdf",
        tmp_path / "models" / "demobioimageio.zip",
        tmp_path / "results" / "test_01.tif",
        tmp_path / "results" / "test_01.zip",
        tmp_path / "results" / "test_01_Nuclei_centre.csv",
        tmp_path / "results" / "Nuclei_count.csv",
        tmp_path / "results" / "test_02_overlay.tif",
        tmp_path / "TrainingDataExample_StarDist2D.png",
    ]
    missing = [str(path.relative_to(tmp_path)) for path in expected if not path.is_file()]
    assert missing == []
