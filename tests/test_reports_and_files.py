"""Reports, figures, provenance and environment information."""

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import synthetic_pair
from stardist2d import config, environment, metrics, plotting, provenance, report


def _environment():
    return environment.EnvironmentInfo({"tensorflow": "2.20.0", "csbdeep": "0.8.2"}, "No GPU found", "No cuda found")


def test_figures_are_written_with_the_1_20_8_names(tmp_path):
    image, labels = synthetic_pair(0)
    plotting.training_pair(image, labels, tmp_path / "TrainingDataExample_StarDist2D.png")
    plotting.loss_curves([3, 2, 1], [3.5, 2.5, 2], tmp_path / "Quality Control" / "lossCurvePlots.png")
    plotting.qc_example(image, labels, labels, 1.0, tmp_path / "Quality Control" / "QC_example_data.png")
    plotting.overlay(image / image.max(), labels, tmp_path / "x_overlay.tif")
    plotting.objects_per_frame([(0, 3), (1, 4)], tmp_path / "plot.png")
    for name in (
        "TrainingDataExample_StarDist2D.png",
        "Quality Control/lossCurvePlots.png",
        "Quality Control/QC_example_data.png",
        "x_overlay.tif",
        "plot.png",
    ):
        assert (tmp_path / name).stat().st_size > 1000


def test_training_and_qc_reports(tmp_path):
    image, labels = synthetic_pair(1)
    figure = tmp_path / "example.png"
    plotting.training_pair(image, labels, figure)
    model_folder = tmp_path / "my_model"
    entry = report.TrainingReport(
        model_folder,
        "1.0.0",
        config.TrainingSettings(),
        config.AugmentationSettings(),
        128,
        40,
        4,
        (128, 128),
        tmp_path / "images",
        tmp_path / "masks",
        "mae",
        False,
        _environment(),
        figure,
        3725.0,
    )
    pdf = report.write_training_report(entry)
    assert pdf.name == "my_model_training_report.pdf" and pdf.read_bytes().startswith(b"%PDF")
    rows = [metrics.qc_row("a.tif", labels, labels), metrics.qc_row("b.tif", labels, np.zeros_like(labels))]
    qc_pdf = report.write_qc_report(model_folder, rows, None, figure)
    assert qc_pdf == model_folder / "Quality Control" / "my_model_QC_report.pdf"
    assert qc_pdf.stat().st_size > 1000


def test_materials_text_is_the_1_20_8_paragraph():
    entry = report.TrainingReport(
        Path("m"),
        "1.0.0",
        config.TrainingSettings(number_of_epochs=7),
        config.AugmentationSettings(),
        256,
        1,
        9,
        (512, 512),
        Path("i"),
        Path("t"),
        "mae",
        True,
        _environment(),
    )
    text = report._materials_text(entry)
    assert text.startswith("The StarDist 2D model was trained for 7 epochs on 9 paired image patches")
    assert "The model was retrained from a pretrained model." in text
    assert "tensorflow (v 2.20.0), csbdeep (v 0.8.2)" in text


def test_run_file_records_settings_inputs_and_versions(tmp_path):
    source = tmp_path / "a.tif"
    source.write_bytes(b"pixels")
    settings = {"training": config.TrainingSettings(), "folder": tmp_path}
    path = provenance.write_run_file(tmp_path / "out", "train", settings, [source], _environment())
    record = provenance.read_run_file(tmp_path / "out")
    assert record["step"] == "train" and record["settings"]["training"]["n_rays"] == 32
    assert record["inputs"][0]["file"] == "a.tif" and len(record["inputs"][0]["sha256"]) == 64
    data = json.loads(path.read_text())
    data["run_file_version"] = 99
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="format version 99"):
        provenance.read_run_file(tmp_path / "out")


def test_environment_reports_versions_and_never_fails():
    info = environment.collect()
    assert info.versions["numpy"] == np.__version__
    assert environment.package_version("not-a-real-package-xyz") == "not installed"
    assert isinstance(environment.gpu_report(), str)
