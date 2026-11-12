"""Training and quality-control PDF reports (1.20.8, cell 8: ``pdf_export`` and ``qc_pdf_export``).

The text is that of 1.20.8. Everything the reports show is passed in (1.20.8 read about 15 notebook
globals), and fpdf2's current API is used (no ``HTMLMixin``, no ``txt=``).
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from stardist2d.config import AugmentationSettings, TrainingSettings
from stardist2d.environment import EnvironmentInfo

if TYPE_CHECKING:
    from fpdf import FPDF

NETWORK = "StarDist 2D"
REFERENCES = (
    'References:\n - ZeroCostDL4Mic: von Chamier, Lucas & Laine, Romain, et al. "Democratising deep learning for '
    'microscopy with ZeroCostDL4Mic." Nature Communications (2021).',
    '- StarDist 2D: Schmidt, Uwe, et al. "Cell detection with star-convex polygons." International Conference on '
    "Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2018.",
)


@dataclass(frozen=True)
class TrainingReport:
    """What the training report shows."""

    model_folder: Path
    notebook_version: str
    settings: TrainingSettings
    augmentation: AugmentationSettings
    patch_size: int
    steps_per_epoch: int
    n_images: int
    image_shape: tuple[int, ...]
    training_source: Path
    training_target: Path
    loss: str
    pretrained: bool
    environment: EnvironmentInfo
    example_figure: Path | None = None
    seconds: float | None = None


def _pdf(header: str) -> FPDF:
    """Return a new A4 page with 1.20.8's bold header."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=11, style="B")
    pdf.multi_cell(180, 5, text=f"{header}\nDate: {datetime.date.today().isoformat()}", align="L")
    pdf.ln(1)
    return pdf


def _line(pdf: FPDF, label: str, value: str, width: int) -> None:
    """Write a bold label followed by its value."""
    pdf.set_font("helvetica", size=10, style="B")
    pdf.cell(width, 5, text=label, align="L")
    pdf.set_font("helvetica", size=10)
    pdf.multi_cell(170, 5, text=value, align="L")
    pdf.ln(1)


def _figure(pdf: FPDF, path: Path | None, x: int, scale: int) -> None:
    """Insert a PNG at 1/*scale* of its pixel size, as 1.20.8 does, when it exists."""
    if path is None or not path.is_file():
        return
    from PIL import Image

    with Image.open(path) as image:
        width, height = image.size
    pdf.image(str(path), x=x, w=round(width / scale), h=round(height / scale))
    pdf.ln(1)


def _materials_text(report: TrainingReport) -> str:
    """Return 1.20.8's "materials and methods" paragraph."""
    origin = "trained" if report.pretrained else "trained from scratch"
    retrained = " The model was retrained from a pretrained model." if report.pretrained else ""
    versions = report.environment.versions
    return (
        f"The {NETWORK} model was {origin} for {report.settings.number_of_epochs} epochs on {report.n_images} paired "
        f"image patches (image dimensions: {report.image_shape}, "
        f"patch size: ({report.patch_size},{report.patch_size})) "
        f"with a batch size of {report.settings.batch_size} and a {report.loss} loss function, using the {NETWORK} "
        f"ZeroCostDL4Mic notebook (v {report.notebook_version}) (von Chamier & Laine et al., 2021).{retrained} "
        "Key python "
        f"packages used include tensorflow (v {versions.get('tensorflow')}), csbdeep (v {versions.get('csbdeep')}), "
        f"cuda (v {report.environment.cuda_version}). The training was accelerated using a "
        f"{report.environment.gpu_name} GPU."
    )


def _parameters_html(report: TrainingReport) -> str:
    """Return the parameter table of the training report."""
    settings = report.settings
    rows = [
        ("number_of_epochs", settings.number_of_epochs),
        ("patch_size", f"{report.patch_size}x{report.patch_size}"),
        ("batch_size", settings.batch_size),
        ("number_of_steps", report.steps_per_epoch),
        ("percentage_validation", settings.percentage_validation),
        ("n_rays", settings.n_rays),
        ("grid_parameter", settings.grid_parameter),
        ("initial_learning_rate", settings.initial_learning_rate),
    ]
    body = "".join(f"<tr><td>{name}</td><td>{value}</td></tr>" for name, value in rows)
    head = '<tr><th width="50%" align="left">Parameter</th><th width="50%" align="left">Value</th></tr>'
    return f'<table width="40%">{head}{body}</table>'


def write_training_report(report: TrainingReport) -> Path:
    """Write ``<model>/<model>_training_report.pdf`` (1.20.8's ``pdf_export``).

    Difference from 1.20.8: the Augmentor reference is no longer cited (1.20.8's augmentation does not use Augmentor).
    """
    name = report.model_folder.name
    pdf = _pdf(f"Training report for {NETWORK} model ({name})")
    if report.seconds is not None:
        minutes, seconds = divmod(report.seconds, 60)
        hours, minutes = divmod(minutes, 60)
        pdf.cell(
            190,
            5,
            text=f"Training time: {int(hours)}hour(s) {int(minutes)}min(s) {round(seconds)}sec(s)",
            new_x="LMARGIN",
            new_y="NEXT",
        )
    pdf.ln(1)
    pdf.cell(190, 5, text="Information for your materials and method:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", size=10)
    pdf.multi_cell(190, 5, text=_materials_text(report), align="L")
    pdf.ln(1)
    factor = report.augmentation.multiply_dataset_by
    augmentation = (
        f"The dataset was augmented by a factor of {factor}"
        if report.augmentation.use_data_augmentation
        else "No augmentation was used for training."
    )
    _line(pdf, "Augmentation: ", augmentation, 28)
    pdf.set_font("helvetica", size=11, style="B")
    pdf.cell(180, 5, text="Parameters", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", size=10)
    if report.settings.use_default_advanced_parameters:
        pdf.cell(200, 5, text="Default Advanced Parameters were enabled", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(200, 5, text="The following parameters were used for training:", new_x="LMARGIN", new_y="NEXT")
    pdf.write_html(_parameters_html(report))
    pdf.set_font("helvetica", size=11, style="B")
    pdf.cell(190, 5, text="Training Dataset", new_x="LMARGIN", new_y="NEXT")
    _line(pdf, "Training_source:", str(report.training_source), 30)
    _line(pdf, "Training_target:", str(report.training_target), 28)
    _line(pdf, "Model Path:", str(report.model_folder), 21)
    pdf.cell(60, 5, text="Example Training pair", new_x="LMARGIN", new_y="NEXT")
    _figure(pdf, report.example_figure, 11, 8)
    _closing(
        pdf,
        "Important:\nRemember to perform the quality control step on all newly trained models\n"
        "Please consider depositing your training dataset on Zenodo",
    )
    return _save(pdf, report.model_folder / f"{name}_training_report.pdf")


def _closing(pdf: FPDF, reminder: str) -> None:
    """Write the references and the bold reminder that end both reports."""
    pdf.set_font("helvetica", size=10)
    for reference in REFERENCES:
        pdf.multi_cell(190, 5, text=reference, align="L")
        pdf.ln(1)
    pdf.ln(3)
    pdf.set_font("helvetica", size=11, style="B")
    pdf.multi_cell(190, 5, text=reminder, align="C")


def _save(pdf: FPDF, path: Path) -> Path:
    """Write *pdf* to *path* (creating its folder)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(path))
    return path


def _metrics_html(rows: list[dict[str, Any]]) -> str:
    """Return the QC metrics table of the QC report (values rounded to 3 decimals, as 1.20.8)."""
    from stardist2d.metrics import QC_COLUMNS

    titles = ["image #", "Prediction v. GT IoU", "false pos.", "true pos.", "false neg.", *QC_COLUMNS[5:]]
    head = "".join(f"<th>{title}</th>" for title in titles)
    rounded = {QC_COLUMNS[i] for i in (1, 5, 6, 7, 8, 11, 12, 13)}
    body = ""
    for index, row in enumerate(rows, start=1):
        cells = [str(index)] + [
            str(round(float(row[column]), 3)) if column in rounded else str(row[column]) for column in QC_COLUMNS[1:]
        ]
        body += "<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"
    return f'<table width="100%"><tr>{head}</tr>{body}</table>'


def write_qc_report(
    model_folder: Path, rows: list[dict[str, Any]], loss_figure: Path | None, example_figure: Path | None
) -> Path:
    """Write ``<model>/Quality Control/<model>_QC_report.pdf`` (1.20.8's ``qc_pdf_export``).

    Difference from 1.20.8: a missing loss-curve figure gives the explanatory sentence instead of an error.
    """
    name = model_folder.name
    pdf = _pdf(f"Quality Control report for {NETWORK} model ({name})")
    pdf.ln(2)
    pdf.cell(190, 5, text="Development of Training Losses", new_x="LMARGIN", new_y="NEXT")
    if loss_figure is not None and loss_figure.is_file():
        _figure(pdf, loss_figure, 11, 8)
    else:
        pdf.set_font("helvetica", size=10)
        pdf.multi_cell(
            190,
            5,
            text="If you would like to see the evolution of the loss function during training "
            "please play the first cell of the QC section in the notebook.",
        )
    pdf.set_font("helvetica", size=10, style="B")
    pdf.ln(3)
    pdf.cell(80, 5, text="Example Quality Control Visualisation", new_x="LMARGIN", new_y="NEXT")
    _figure(pdf, example_figure, 16, 10)
    pdf.set_font("helvetica", size=11, style="B")
    pdf.cell(180, 5, text="Quality Control Metrics", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", size=7)
    pdf.write_html(_metrics_html(rows))
    _closing(
        pdf,
        "To find the parameters and other information about how this model was trained, go to the "
        "training_report.pdf of this model which should be in the folder of the same name.",
    )
    return _save(pdf, model_folder / "Quality Control" / f"{name}_QC_report.pdf")
