"""Predict with a StarDist 2D model (1.20.8, cells 41 and 43): single images, stacks, and very large images.

Prediction runs through StarDist's tile-by-tile generator, so a caller (the LabConstrictor-Tools
bridge, the notebook) can show progress and cancel between tiles.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from csbdeep.data import Normalizer, normalize_mi_ma

from stardist2d import prepare

if TYPE_CHECKING:
    from stardist.models import StarDist2D


def _no_progress(fraction: float, message: str) -> None:
    """Ignore a progress report."""
    _ = fraction, message


def _never_cancel() -> None:
    """Never cancel."""


@dataclass(frozen=True)
class Hooks:
    """Callbacks a caller passes in: ``progress(fraction, message)`` and ``check_cancel()`` (raises to stop)."""

    progress: Callable[[float, str], None] = field(default=_no_progress)
    check_cancel: Callable[[], None] = field(default=_never_cancel)


@dataclass(frozen=True)
class Detection:
    """Prediction settings. ``None`` thresholds use the model's optimised values (``thresholds.json``).

    ``n_tiles=None`` predicts the image in one piece, as 1.20.8; ``"auto"`` lets StarDist choose tiles
    from the image size (``_guess_n_tiles``), which keeps memory low on large images.
    """

    prob_thresh: float | None = None
    nms_thresh: float | None = None
    low_percentile: float = prepare.LOW_PERCENTILE
    high_percentile: float = prepare.HIGH_PERCENTILE
    n_tiles: tuple[int, ...] | str | None = None


def _tiles(model: StarDist2D, image: np.ndarray, n_tiles: tuple[int, ...] | str | None) -> tuple[int, ...] | None:
    """Return the tiles to use for *image* (see :class:`Detection`)."""
    if n_tiles == "auto":
        return tuple(model._guess_n_tiles(image))
    return n_tiles if n_tiles is None or isinstance(n_tiles, tuple) else None


def predict_normalised(
    model: StarDist2D, image: np.ndarray, detection: Detection | None = None, hooks: Hooks | None = None
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return ``(labels, details)`` for an image that is already normalised.

    ``details`` holds StarDist's ``coord`` (polygons), ``points`` (centres) and ``prob`` (scores).
    """
    detection, hooks = detection or Detection(), hooks or Hooks()
    tiles = _tiles(model, image, detection.n_tiles)
    total = int(np.prod(tiles)) if tiles else 1
    done = 0
    result: Any = None
    for step in model._predict_instances_generator(
        image,
        prob_thresh=detection.prob_thresh,
        nms_thresh=detection.nms_thresh,
        n_tiles=tiles,
        show_tile_progress=False,
    ):
        hooks.check_cancel()
        if step == "tile":
            done += 1
            hooks.progress(0.9 * done / total, f"predicted tile {done} of {total}")
        elif step == "nms":
            hooks.progress(0.9, "separating the objects")
        elif not isinstance(step, str):
            result = step
    labels, details = result
    hooks.progress(1.0, f"{len(details['coord'])} objects")
    return labels, details


def predict_image(
    model: StarDist2D, image: np.ndarray, detection: Detection | None = None, hooks: Hooks | None = None
) -> tuple[np.ndarray, dict[str, Any]]:
    """Normalise a YX or YXC image as 1.20.8 does for prediction (cell 41) and return ``(labels, details)``."""
    detection = detection or Detection()
    normalised = prepare.normalize_image(
        image, prepare.prediction_axes(image), detection.low_percentile, detection.high_percentile
    )
    return predict_normalised(model, normalised, detection, hooks)


def stack_percentiles(stack: np.ndarray, low: float, high: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the *low* and *high* percentiles of a TYX stack over all frames (cell 41, stacks)."""
    axes = (0, 1, 2)
    return (
        np.percentile(stack, low, axis=axes, keepdims=True)[0],
        np.percentile(stack, high, axis=axes, keepdims=True)[0],
    )


def normalise_frame(frame: np.ndarray, low_value: np.ndarray, high_value: np.ndarray) -> np.ndarray:
    """Return *frame* scaled so that the stack's low percentile becomes 0 and its high percentile 1.

    Difference from 1.20.8: 1.20.8 divided by ``(low - high)``, which made every frame negative.
    """
    return ((frame - low_value) / (high_value - low_value + 1e-20)).astype(np.float32)


def big_image_range(image: np.ndarray, low: float, high: float) -> tuple[float, float]:
    """Return the intensity range used to normalise a very large image (cell 43).

    Difference from 1.20.8: 1.20.8 used 0 and 255 for every image, wrong for 16-bit or float images.
    The percentiles are taken on every 8th pixel, the option 1.20.8 left commented out.
    """
    sample = image[::8, ::8] if image.ndim >= 2 else image
    low_value, high_value = np.percentile(sample, [low, high])
    return float(low_value), float(high_value)


def predict_big(
    model: StarDist2D, image: np.ndarray, block_size: int, min_overlap: int, n_tiles: tuple[int, int], workdir: Path
) -> tuple[np.ndarray, dict[str, Any]]:
    """Predict a very large YX or YXC image block by block through zarr (cell 43, ``predict_img``)."""
    import zarr

    low_value, high_value = big_image_range(image, prepare.LOW_PERCENTILE, prepare.HIGH_PERCENTILE)
    normalizer = _FixedRangeNormalizer(low_value, high_value)
    if image.ndim == 2:
        axes, blocks, overlap, tiles = "YX", (block_size, block_size), (min_overlap, min_overlap), n_tiles
    else:
        axes, blocks = "YXC", (block_size, block_size, image.shape[-1])
        overlap, tiles = (min_overlap, min_overlap, 0), (*n_tiles, 1)
    workdir.mkdir(parents=True, exist_ok=True)
    zarr.save_array(str(workdir / "image.zarr"), image)
    stored = zarr.open(str(workdir / "image.zarr"), mode="r")
    return model.predict_instances_big(
        stored,
        axes=axes,
        block_size=blocks,
        min_overlap=overlap,
        context=None,
        normalizer=normalizer,
        show_progress=False,
        n_tiles=tiles,
    )


class _FixedRangeNormalizer(Normalizer):
    """A CSBDeep normaliser with a fixed intensity range (1.20.8's ``MyNormalizer``, with a working ``after``)."""

    def __init__(self, low_value: float, high_value: float) -> None:
        self.low_value, self.high_value = low_value, high_value

    @property
    def do_after(self) -> bool:
        """Return false: the prediction is never mapped back to the input range."""
        return False

    def before(self, x: np.ndarray, axes: str) -> np.ndarray:
        """Return *x* scaled to the fixed range."""
        _ = axes
        return normalize_mi_ma(x, self.low_value, self.high_value, dtype=np.float32)

    def after(self, mean: np.ndarray, scale: np.ndarray | None, axes: str) -> None:
        """Refuse: a fixed-range normaliser is never undone (StarDist does not call it)."""
        _ = mean, scale, axes
        raise NotImplementedError("the fixed-range normaliser cannot be undone")


@dataclass(frozen=True)
class FolderResult:
    """What :func:`predict_folder` wrote: one ``(file name, number of objects)`` per image, and the last prediction."""

    counts: list[tuple[str, int]]
    last_image: np.ndarray
    last_labels: np.ndarray


def predict_folder(
    model: StarDist2D,
    images_folder: Path,
    results_folder: Path,
    detection: Detection | None = None,
    export_rois: bool = True,
    hooks: Hooks | None = None,
) -> FolderResult:
    """Predict every ``*.tif`` image of a folder and write 1.20.8's single-image outputs (cell 41).

    Per image: ``<name>.tif`` (labels), ``<stem>.zip`` (ImageJ ROIs, when *export_rois*) and
    ``<stem>_Nuclei_centre.csv``; for the folder: ``Nuclei_count.csv`` (no header, as 1.20.8).

    Difference from 1.20.8: the ROI choice is honoured (1.20.8 always wrote the ROI zip whatever the
    "Region_of_interests" box said); the notebook's box now defaults to on.
    """
    from stardist2d import data, export

    detection, hooks = detection or Detection(), hooks or Hooks()
    files = data.image_files(images_folder, "prediction images")
    results_folder.mkdir(parents=True, exist_ok=True)
    counts: list[tuple[str, int]] = []
    image = labels = np.zeros((0, 0))
    for index, path in enumerate(files):
        hooks.progress(index / len(files), f"predicting {path.name}")
        image = data.read_image(path)
        labels, details = predict_image(model, image, detection, Hooks(check_cancel=hooks.check_cancel))
        export.write_labels(results_folder / path.name, labels)
        if export_rois:
            export.write_imagej_rois(results_folder / f"{path.stem}.zip", details["coord"])
        export.write_centres(results_folder / f"{path.stem}_Nuclei_centre.csv", details["points"])
        counts.append((path.name, int(details["coord"].shape[0])))
    export.write_counts(results_folder / "Nuclei_count.csv", counts)
    hooks.progress(1.0, f"{len(files)} images predicted")
    return FolderResult(counts, image, labels)
