"""What the Fiji, napari and QuPath tools call (``stardist2d_lc_tools``, LabConstrictor-Tools bridge).

``labconstrictor_tools`` is imported only inside the functions: the package must work without it
(Colab, the notebook, the command line). Loaded models are cached by their settings, so a host that
keeps its worker (napari, QuPath) does not reload TensorFlow and the weights at every run.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from stardist2d import data, metrics, model, predict

if TYPE_CHECKING:
    import pandas as pd
    from labconstrictor_tools.diagnostics import Check
    from stardist.models import StarDist2D

MAX_LABELS = 65535
"""More objects cannot be sent back as a 16-bit label image (hosts would get a float image)."""


@dataclass(frozen=True)
class ModelChoice:
    """The model a tool's form names: a pretrained name, or ``Custom model folder`` with the folder."""

    source: str
    folder: Path | None = None


@dataclass(frozen=True)
class DetectionSettings:
    """Thresholds (``None``: the model's optimised values) and the normalisation percentiles."""

    prob_thresh: float | None = None
    nms_thresh: float | None = None
    low_percentile: float = 1.0
    high_percentile: float = 99.8


def _tool_error(code: str, text: str) -> Exception:
    """Return a LabConstrictor-Tools ``ToolError`` (shown to the person as a plain message)."""
    from labconstrictor_tools import ToolError

    return ToolError(code, text)


@functools.lru_cache(maxsize=2)
def _cached_model(source: str, folder: str, modified: float) -> StarDist2D:
    """Load a model once per (source, folder, modification time)."""
    _ = modified
    return model.load_model(model.ModelChoice(source, Path(folder) if folder else None))


def load(choice: ModelChoice) -> StarDist2D:
    """Return the model of *choice*, loaded once; refuse a missing or wrong folder with what to do."""
    try:
        resolved = model.ModelChoice(choice.source, choice.folder)
    except ValueError as problem:
        raise _tool_error("invalid_parameter", str(problem)) from None
    folder = str(resolved.folder or "")
    modified = (
        (Path(folder) / "config.json").stat().st_mtime if folder and (Path(folder) / "config.json").is_file() else 0
    )
    try:
        return _cached_model(resolved.source, folder, modified)
    except FileNotFoundError as problem:
        raise _tool_error("invalid_parameter", str(problem)) from None


def _hooks() -> predict.Hooks:
    """Return hooks that report progress to the host and stop when the person cancels."""
    from labconstrictor_tools import check_cancel, progress

    return predict.Hooks(progress=progress, check_cancel=check_cancel)


def check_image(image: np.ndarray, star_dist: StarDist2D) -> np.ndarray:
    """Return *image* if its layout fits the model (YX for one channel, YX3 RGB for three), else refuse.

    The hosts only check the number of dimensions, so a Z-stack or a two-channel image would otherwise reach the model.
    """
    channels = star_dist.config.n_channel_in
    if channels == 1 and image.ndim != 2:
        raise _tool_error("wrong_dimensions", f"this model needs a single-channel 2D image; got shape {image.shape}")
    if channels > 1 and (image.ndim != 3 or image.shape[-1] != channels):
        raise _tool_error(
            "wrong_dimensions", f"this model needs a 2D image with {channels} channels (RGB); got shape {image.shape}"
        )
    return image


def _crop(region: np.ndarray | None, image: np.ndarray) -> tuple[slice, slice]:
    """Return the rows and columns to predict: the bounding box of the selected region, or the whole image."""
    if region is None:
        return slice(0, image.shape[0]), slice(0, image.shape[1])
    from labconstrictor_tools.region import bbox

    return bbox(region, image)


def _objects_table(labels: np.ndarray, details: dict[str, Any]) -> pd.DataFrame:
    """Return one row per object: label, centre (y, x), area in pixels and StarDist's probability."""
    import pandas as pd

    count = len(details["points"])
    areas = np.bincount(labels.ravel(), minlength=count + 1)[1 : count + 1]
    points = np.asarray(details["points"], dtype=float).reshape(-1, 2)
    return pd.DataFrame(
        {
            "label": np.arange(1, count + 1),
            "centroid_y": points[:, 0],
            "centroid_x": points[:, 1],
            "area_px": areas,
            "probability": np.asarray(details["prob"], dtype=float),
        }
    )


def _shapes(details: dict[str, Any], offset: tuple[int, int]) -> list[dict[str, Any]]:
    """Return StarDist's own polygons (the notebook's ImageJ ROIs) as ``{polygon: (y, x) vertices, label, prob}``."""
    shift = np.array(offset, dtype=float)
    return [
        {"polygon": np.asarray(coord, dtype=float).T + shift, "label": index, "prob": float(probability)}
        for index, (coord, probability) in enumerate(zip(details["coord"], details["prob"], strict=True), start=1)
    ]


def segment(
    image: np.ndarray, choice: ModelChoice, settings: DetectionSettings, region: np.ndarray | None = None
) -> tuple[np.ndarray, list[dict[str, Any]], pd.DataFrame, str]:
    """Segment one image: labels (16-bit), outlines, per-object table and a one-line summary.

    With a *region*, only its bounding box is predicted; objects whose centre lies outside the region are dropped.
    """
    star_dist = load(choice)
    image = check_image(np.asarray(image), star_dist)
    rows, columns = _crop(region, image)
    detection = predict.Detection(
        settings.prob_thresh, settings.nms_thresh, settings.low_percentile, settings.high_percentile, "auto"
    )
    labels, details = predict.predict_image(star_dist, image[rows, columns], detection, _hooks())
    if region is not None:
        labels, details = _inside(labels, details, np.asarray(region)[rows, columns] > 0)
    if len(details["points"]) == 0:
        raise _tool_error("no_match", "No object found: try a lower probability threshold, or another model.")
    if len(details["points"]) > MAX_LABELS:
        raise _tool_error(
            "too_many_objects", f"{len(details['points'])} objects: select a smaller region (at most {MAX_LABELS})."
        )
    full = np.zeros(image.shape[:2], dtype=np.uint16)
    full[rows, columns] = labels
    offset = (rows.start or 0, columns.start or 0)
    table = _objects_table(full, details)
    table["centroid_y"] += offset[0]
    table["centroid_x"] += offset[1]
    return full, _shapes(details, offset), table, f"Found **{len(table)}** objects."


def _inside(labels: np.ndarray, details: dict[str, Any], mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep the objects whose centre is inside *mask* and renumber them 1..n."""
    points = np.asarray(details["points"]).reshape(-1, 2).astype(int)
    keep = np.flatnonzero(mask[points[:, 0], points[:, 1]]) if len(points) else np.array([], int)
    lookup = np.zeros(len(points) + 1, dtype=np.int32)
    lookup[keep + 1] = np.arange(1, len(keep) + 1)
    kept = {key: np.asarray(details[key])[keep] for key in ("coord", "points", "prob")}
    return lookup[labels], kept


def segment_folder(
    input_folder: Path, output_folder: Path, choice: ModelChoice, settings: DetectionSettings, export_rois: bool
) -> tuple[pd.DataFrame, str]:
    """Segment every ``*.tif`` of a folder with the notebook's section 6.1 outputs; return the counts table."""
    import pandas as pd

    if Path(output_folder).resolve() == Path(input_folder).resolve():
        raise _tool_error("invalid_parameter", "choose an output folder other than the input folder")
    detection = predict.Detection(
        settings.prob_thresh, settings.nms_thresh, settings.low_percentile, settings.high_percentile, "auto"
    )
    result = predict.predict_folder(
        load(choice), Path(input_folder), Path(output_folder), detection, export_rois, _hooks()
    )
    table = pd.DataFrame(result.counts, columns=["image", "objects"])
    return table, f"Segmented **{len(table)}** images into {output_folder}."


def quality_control(
    image: np.ndarray, ground_truth: np.ndarray, choice: ModelChoice, iou_threshold: float
) -> tuple[np.ndarray, pd.DataFrame, str]:
    """Predict *image* and compare with *ground_truth* (the notebook's section 5.2 metrics)."""
    import pandas as pd

    star_dist = load(choice)
    labels, _ = predict.predict_image(
        star_dist, check_image(np.asarray(image), star_dist), predict.Detection(n_tiles="auto"), _hooks()
    )
    row = metrics.qc_row("image", np.asarray(ground_truth), labels, iou_threshold)
    summary = f"F1 **{row['f1 score']:.3f}**, IoU {row['Prediction v. GT Intersection over Union']:.3f}"
    return labels.astype(np.uint16), pd.DataFrame([row]), summary


def tensorflow_checks() -> list[Check]:
    """Return a diagnostics line for TensorFlow: version, Keras 2 and the devices it sees."""
    from labconstrictor_tools.diagnostics import Check

    from stardist2d.keras_backend import check_keras_backend

    check_keras_backend()
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    detail = f"TensorFlow {tf.__version__}, GPUs: {len(gpus)}"
    if gpus:
        return [Check("gpu libraries", "TensorFlow", "ok", detail)]
    return [
        Check(
            "gpu libraries",
            "TensorFlow",
            "info",
            detail,
            "Prediction runs on the CPU; train on a GPU machine or Colab.",
        )
    ]


def example_folder() -> Path | None:
    """Return the example images shipped with the app (for the tools' test cases)."""
    return data.bundled_example_data()
