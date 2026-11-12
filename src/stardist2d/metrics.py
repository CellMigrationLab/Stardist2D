"""Quality-control metrics of notebook section 5.2 (ZeroCostDL4Mic 1.20.8, cell 36).

Pure: arrays in, values out. One row per test image, with the columns of 1.20.8's
``Quality_Control for <model>.csv``, in the same order.
"""

from __future__ import annotations

from typing import Any

import numpy as np

QC_COLUMNS = (
    "image",
    "Prediction v. GT Intersection over Union",
    "false positive",
    "true positive",
    "false negative",
    "precision",
    "recall",
    "accuracy",
    "f1 score",
    "n_true",
    "n_pred",
    "mean_true_score",
    "mean_matched_score",
    "panoptic_quality",
)
"""Columns of the quality-control table, exactly as 1.20.8 writes them."""

_MATCHING_FIELDS = (
    "fp",
    "tp",
    "fn",
    "precision",
    "recall",
    "accuracy",
    "f1",
    "n_true",
    "n_pred",
    "mean_true_score",
    "mean_matched_score",
    "panoptic_quality",
)

DEFAULT_IOU_THRESHOLD = 0.5
"""IoU above which a predicted object matches a ground-truth object (1.20.8 uses 0.5)."""


def binary_iou(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    """Return the intersection over union of the foreground (label > 0) of two label images.

    Two empty images give ``nan``, as 1.20.8 does (0 / 0), without a runtime warning.
    """
    truth, predicted = ground_truth > 0, prediction > 0
    union = int(np.logical_or(truth, predicted).sum())
    if union == 0:
        return float("nan")
    return int(np.logical_and(truth, predicted).sum()) / union


def qc_row(
    image_name: str, ground_truth: np.ndarray, prediction: np.ndarray, iou_threshold: float = DEFAULT_IOU_THRESHOLD
) -> dict[str, Any]:
    """Return the quality-control row of one image: foreground IoU and StarDist's object matching.

    Parameters
    ----------
    image_name
        File name written in the ``image`` column.
    ground_truth, prediction
        Label images of the same shape.
    iou_threshold
        Object IoU above which a predicted object counts as a true positive.

    Raises
    ------
    ValueError
        If the two label images have different shapes.
    """
    if ground_truth.shape != prediction.shape:
        raise ValueError(
            f"{image_name}: the ground truth {ground_truth.shape} and the prediction {prediction.shape} differ in shape"
        )
    from stardist.matching import matching

    stats = matching(ground_truth, prediction, thresh=iou_threshold)
    values = [image_name, binary_iou(ground_truth, prediction), *(getattr(stats, field) for field in _MATCHING_FIELDS)]
    return dict(zip(QC_COLUMNS, values, strict=True))
