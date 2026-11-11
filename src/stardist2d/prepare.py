"""Prepare training and prediction images as notebook sections 4.1, 5.2 and 6.1 do (1.20.8, cells 28, 36, 41).

Pure: arrays in, arrays out. Normalisation is CSBDeep's percentile normalisation (1 and 99.8
percent), the one StarDist and 1.20.8 use.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

LOW_PERCENTILE = 1.0
HIGH_PERCENTILE = 99.8
SPLIT_SEED = 42
"""1.20.8 splits training and validation images with ``np.random.RandomState(42)``."""


@dataclass(frozen=True)
class TrainingSplit:
    """Normalised training and validation images and their hole-filled label images."""

    x_train: list[np.ndarray]
    y_train: list[np.ndarray]
    x_val: list[np.ndarray]
    y_val: list[np.ndarray]


def channel_count(image: np.ndarray) -> int:
    """Return the number of channels of a YX or YXC image (cell 28: 1 for a 2D image, else the last axis)."""
    return 1 if image.ndim == 2 else int(image.shape[-1])


def drop_alpha(images: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Return the images without the alpha channel when they are RGBA (cell 28), unchanged otherwise."""
    if images and channel_count(images[0]) == 4:
        return [image[:, :, :3] for image in images]
    return list(images)


def training_axes(image: np.ndarray) -> tuple[int, ...]:
    """Return the normalisation axes used for *training* (cell 28): each channel on its own, ``(0, 1)``."""
    _ = image
    return (0, 1)


def prediction_axes(image: np.ndarray) -> tuple[int, ...]:
    """Return the normalisation axes used for *quality control and prediction* (cells 36, 41).

    One channel: ``(0, 1)``. Several channels: ``(0, 1, 2)``, all channels together. 1.20.8 trains with
    channels normalised separately but predicts with them normalised together; this port keeps both as
    they are (see docs/plan.md, open scientific question).
    """
    return (0, 1) if channel_count(image) == 1 else (0, 1, 2)


def normalize_image(
    image: np.ndarray, axes: tuple[int, ...], low: float = LOW_PERCENTILE, high: float = HIGH_PERCENTILE
) -> np.ndarray:
    """Return *image* scaled so that its *low* and *high* percentiles (over *axes*) become 0 and 1."""
    from csbdeep.utils import normalize

    return normalize(image, low, high, axis=axes)


def fill_holes(labels: np.ndarray) -> np.ndarray:
    """Return *labels* with the holes inside each object filled (``stardist.fill_label_holes``)."""
    from stardist import fill_label_holes

    return fill_label_holes(labels)


def split_training_data(
    images: Sequence[np.ndarray], labels: Sequence[np.ndarray], n_validation: int
) -> TrainingSplit:
    """Return the images split as 1.20.8 does: a ``RandomState(42)`` permutation, last *n_validation* for validation.

    Raises
    ------
    ValueError
        If the numbers of images and labels differ, or no image would be left for training.
    """
    if len(images) != len(labels):
        raise ValueError(f"{len(images)} images but {len(labels)} label images")
    if not 1 <= n_validation < len(images):
        raise ValueError(f"cannot keep {n_validation} of {len(images)} images for validation and train on the rest")
    order = np.random.RandomState(SPLIT_SEED).permutation(len(images))
    train, val = order[:-n_validation], order[-n_validation:]
    return TrainingSplit(
        [images[i] for i in train], [labels[i] for i in train], [images[i] for i in val], [labels[i] for i in val]
    )
