"""Quality control of a trained model (1.20.8, section 5, cells 32, 34 and 36)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from stardist2d import data, export, metrics, predict

if TYPE_CHECKING:
    from stardist.models import StarDist2D

QC_FOLDER = "Quality Control"


@dataclass(frozen=True)
class QualityControl:
    """The rows of the QC table and where the table and the predictions were written."""

    rows: list[dict[str, Any]]
    table: Path
    predictions: Path


def read_training_evaluation(model_folder: Path) -> tuple[list[float], list[float]]:
    """Return the training and validation losses of ``training_evaluation.csv`` (cell 34).

    Raises
    ------
    FileNotFoundError
        If the model has no ``Quality Control/training_evaluation.csv`` (it was not trained in this notebook).
    """
    path = model_folder / QC_FOLDER / "training_evaluation.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist: the loss curves need a model trained with this notebook")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))[1:]
    return [float(row[0]) for row in rows], [float(row[1]) for row in rows]


def run_quality_control(
    star_dist: StarDist2D,
    model_folder: Path,
    images_folder: Path,
    masks_folder: Path,
    iou_threshold: float = metrics.DEFAULT_IOU_THRESHOLD,
    hooks: predict.Hooks | None = None,
) -> QualityControl:
    """Predict the QC images, compare them with their masks and write 1.20.8's QC outputs (cell 36).

    Writes ``Quality Control/Prediction/<name>.tif`` and ``Quality Control/Quality_Control for <model>.csv``.
    """
    hooks = hooks or predict.Hooks()
    pairs = data.training_pairs(images_folder, masks_folder)
    predictions = model_folder / QC_FOLDER / "Prediction"
    rows = []
    for index, (image_path, mask_path) in enumerate(pairs):
        hooks.progress(index / len(pairs), f"quality control of {image_path.name}")
        labels, _ = predict.predict_image(
            star_dist, data.read_image(image_path), hooks=predict.Hooks(check_cancel=hooks.check_cancel)
        )
        export.write_labels(predictions / image_path.name, labels)
        rows.append(metrics.qc_row(image_path.name, data.read_image(mask_path), labels, iou_threshold))
    table = export.write_qc_table(model_folder / QC_FOLDER / f"Quality_Control for {model_folder.name}.csv", rows)
    _write_undefined_values(table, rows, hooks)
    hooks.progress(1.0, f"{len(rows)} images evaluated")
    return QualityControl(rows, table, predictions)


def _write_undefined_values(table: Path, rows: list[dict[str, Any]], hooks: predict.Hooks) -> Path | None:
    """Explain every NaN of the QC table in ``<table>_undefined.csv`` (image, column, reason); report how many.

    The QC table keeps 1.20.8's columns and its NaN (two empty masks have no IoU); the reason goes beside it.
    """
    notes = [(row["image"], column, reason) for row in rows for column, reason in metrics.undefined_values(row)]
    if not notes:
        return None
    path = table.with_name(f"{table.stem}_undefined.csv")
    export.write_csv(path, [("image", "column", "reason"), *notes], "\n")
    hooks.progress(1.0, f"{len(notes)} undefined value(s) in the QC table: reasons in {path.name}")
    return path
