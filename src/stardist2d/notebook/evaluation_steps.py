"""Notebook section 5: model choice, loss curves, quality control, BioImage Model Zoo export (1.20.8 cells 32–38)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import ipywidgets as widgets
from IPython.display import display

from stardist2d import bioimage, data, export, model, plotting, qc, report
from stardist2d.notebook.forms import Field, Form, message
from stardist2d.notebook.session import Session


def choose_qc_model(session: Session) -> Form:
    """Section 5: the model to evaluate, the one just trained or a folder (cell 32)."""

    def run(values: dict[str, Any]) -> None:
        session.qc_model_folder = session.model_from_choice(
            values["Use_the_current_trained_model"], values["QC_model_folder"]
        )
        message(f"The {session.qc_model_folder.name} network will be evaluated")

    fields = [Field("Use_the_current_trained_model", True, "bool"), Field("QC_model_folder", "")]
    return Form("Model to evaluate", fields, run).show()


def _qc_folder(session: Session) -> Path:
    """Return the model being evaluated; refuse when section 5 was not run."""
    if session.qc_model_folder is None:
        raise ValueError("run the first cell of section 5 first")
    return session.qc_model_folder


def loss_curves(session: Session) -> Form:
    """Section 5.1: training and validation loss against epochs (cell 34, ``lossCurvePlots.png``)."""

    def run(values: dict[str, Any]) -> None:
        _ = values
        folder = _qc_folder(session)
        loss, val_loss = qc.read_training_evaluation(folder)
        display(plotting.loss_curves(loss, val_loss, folder / qc.QC_FOLDER / "lossCurvePlots.png"))

    return Form("Plot of training errors vs. epoch number", [], run, "Show").show()


def _viewer(session: Session) -> None:
    """Show the per-image QC viewer of cell 36 (a dropdown of the test images) and save its first view."""
    folder = _qc_folder(session)
    if session.qc_images is None or session.qc_masks is None:
        raise ValueError("run section 5.2 first")
    images, masks = session.qc_images, session.qc_masks
    names = [row["image"] for row in session.qc_rows]
    choose = widgets.Dropdown(options=names, description="file")
    area = widgets.Output()

    def show(name: str) -> None:
        row = next(item for item in session.qc_rows if item["image"] == name)
        image, truth = data.read_image(images / name), data.read_image(masks / name)
        prediction = data.read_image(folder / qc.QC_FOLDER / "Prediction" / name)
        figure = plotting.qc_example(
            image,
            truth,
            prediction,
            row["Prediction v. GT Intersection over Union"],
            folder / qc.QC_FOLDER / "QC_example_data.png",
        )
        area.clear_output()
        with area:
            display(figure)

    choose.observe(lambda change: show(change["new"]), names="value")
    display(widgets.VBox([choose, area]))
    if names:
        show(names[0])


def quality_control(session: Session) -> Form:
    """Section 5.2: predict the QC images, metrics table, viewer and QC report (cell 36)."""

    def run(values: dict[str, Any]) -> None:
        folder = _qc_folder(session)
        session.qc_images, session.qc_masks = (
            session.path(values["Source_QC_folder"]),
            session.path(values["Target_QC_folder"]),
        )
        result = qc.run_quality_control(model.load_model_folder(folder), folder, session.qc_images, session.qc_masks)
        session.qc_rows = result.rows
        export.print_table(result.rows, sys.stdout)
        _viewer(session)
        loss = folder / qc.QC_FOLDER / "lossCurvePlots.png"
        report.write_qc_report(folder, result.rows, loss, folder / qc.QC_FOLDER / "QC_example_data.png")
        message(f"Quality control report written to {folder / qc.QC_FOLDER}")

    fields = [
        Field("Source_QC_folder", "data/Stardist/Test - Images"),
        Field("Target_QC_folder", "data/Stardist/Test - Masks"),
    ]
    return Form("Error mapping and quality metrics estimation", fields, run).show()


def export_bioimage_model_zoo(session: Session) -> Form:
    """Section 5.3: export the evaluated model in the BioImage Model Zoo format (cell 38)."""

    def run(values: dict[str, Any]) -> None:
        folder = _qc_folder(session)
        info = bioimage.ModelMetadata(
            values["Trained_model_name"] or folder.name,
            bioimage.parse_list(values["Trained_model_authors"]),
            bioimage.parse_list(values["Trained_model_authors_affiliation"]),
            values["Trained_model_description"],
            values["Trained_model_license"],
        )
        if values["default_example_image"]:
            if session.qc_images is None:
                raise ValueError("run section 5.2 first, or untick default_example_image and give fileID")
            example = data.read_image(data.image_files(session.qc_images)[0])
        else:
            example = data.read_image(session.path(values["fileID"]))
        output = bioimage.export_model(model.load_model_folder(folder), bioimage.default_output(folder), example, info)
        message(f"The model was exported to {output}")

    fields = [
        Field("Trained_model_name", ""),
        Field("Trained_model_authors", "[Author 1, Author 2, Author 3]"),
        Field("Trained_model_authors_affiliation", "[Affiliation 1, Affiliation 2, Affiliation 3]"),
        Field("Trained_model_description", ""),
        Field("Trained_model_license", "MIT"),
        Field("PixelSize", 1.0, "float", help="Pixel size in microns of the example image (recorded, as in 1.20.8)"),
        Field("default_example_image", True, "bool"),
        Field("fileID", ""),
    ]
    return Form("Export your model into the BioImage Model Zoo format", fields, run, "Export").show()
