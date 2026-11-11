"""Stacks and very large images (1.20.8, section 6.1 "Stacks" and section 6.2), folder by folder.

Both can resume: frames already written in a stack's result folder are not predicted again.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from stardist2d import data, export, predict

if TYPE_CHECKING:
    from stardist.models import StarDist2D


@dataclass(frozen=True)
class StackResult:
    """Per stack: ``(frame index, number of objects)`` for the frames predicted in this run."""

    counts: dict[str, list[tuple[int, int]]]
    skipped: tuple[str, ...]


def _frame_name(index: int) -> str:
    """Return 1.20.8's frame file name (``frame_00012.tif``)."""
    return f"frame_{str(index).zfill(5)}.tif"


def _first_missing_frame(folder: Path) -> int:
    """Return how many frames *folder* already holds (0 when it does not exist), as 1.20.8 counts them."""
    return len(list(folder.iterdir())) if folder.is_dir() else 0


def predict_stack(
    model: StarDist2D, path: Path, results_folder: Path, export_rois: bool, hooks: predict.Hooks
) -> list[tuple[int, int]] | None:
    """Predict a TYX stack frame by frame and write 1.20.8's stack outputs (cell 41, stacks).

    Writes ``<stem>/frame_XXXXX.tif`` (labels), ``<stem>_Nuclei_number.csv`` (no header) and, when
    *export_rois*, one ``<stem>.zip`` with the ROIs of every frame (each ROI carries its frame position).
    Returns ``None`` when every frame was already written.

    Differences from 1.20.8: frames are normalised correctly (:func:`stardist2d.predict.normalise_frame`);
    labels are written as integers (1.20.8 converted them to 32-bit float); the ROIs come from the same
    prediction as the labels (1.20.8 predicted every frame a second time, without normalisation). When a
    stack is resumed, the ROI zip is not written, because the frames predicted earlier have no ROIs.
    """
    stack = data.read_image(path)
    frames_folder = results_folder / path.stem
    start = _first_missing_frame(frames_folder)
    if start >= stack.shape[0]:
        return None
    low_value, high_value = predict.stack_percentiles(
        stack, predict.prepare.LOW_PERCENTILE, predict.prepare.HIGH_PERCENTILE
    )
    frames_folder.mkdir(parents=True, exist_ok=True)
    counts, polygons = [], []
    for index in range(start, stack.shape[0]):
        hooks.check_cancel()
        hooks.progress(index / stack.shape[0], f"{path.name}: frame {index + 1} of {stack.shape[0]}")
        frame = predict.normalise_frame(stack[index], low_value, high_value)
        labels, details = predict.predict_normalised(model, frame, hooks=predict.Hooks(check_cancel=hooks.check_cancel))
        export.write_labels(frames_folder / _frame_name(index), labels)
        counts.append((index, int(details["coord"].shape[0])))
        polygons.append(details["coord"])
    export.write_counts(results_folder / f"{path.stem}_Nuclei_number.csv", [(str(i), n) for i, n in counts])
    if export_rois and start == 0:
        export.write_imagej_rois(results_folder / f"{path.stem}.zip", polygons)
    return counts


def predict_stacks(
    model: StarDist2D,
    images_folder: Path,
    results_folder: Path,
    export_rois: bool = True,
    hooks: predict.Hooks | None = None,
) -> StackResult:
    """Predict every ``*.tif`` stack of a folder (:func:`predict_stack`)."""
    hooks = hooks or predict.Hooks()
    results_folder.mkdir(parents=True, exist_ok=True)
    counts: dict[str, list[tuple[int, int]]] = {}
    skipped: list[str] = []
    for path in data.image_files(images_folder, "stacks"):
        result = predict_stack(model, path, results_folder, export_rois, hooks)
        if result is None:
            skipped.append(path.name)
        else:
            counts[path.name] = result
    return StackResult(counts, tuple(skipped))


@dataclass(frozen=True)
class BigImageSettings:
    """Section 6.2 settings (names and defaults of 1.20.8)."""

    block_size: int = 512
    min_overlap: int = 50
    n_tiles: tuple[int, int] = (1, 1)
    is_stack: bool = True
    mask_images: bool = True
    region_of_interests: bool = True

    def __post_init__(self) -> None:
        if self.block_size <= 2 * self.min_overlap:
            raise ValueError("block_size must be larger than twice min_overlap")
        if not (self.mask_images or self.region_of_interests):
            raise ValueError("choose at least one output: mask images or regions of interest")


def _write_big_outputs(
    labels: np.ndarray, coord: np.ndarray, mask: Path, rois: Path, settings: BigImageSettings
) -> None:
    """Write the label image and/or ROI zip of one big image or frame (cell 43 names)."""
    if settings.mask_images:
        export.write_labels(mask, np.asarray(labels))
    if settings.region_of_interests:
        export.write_imagej_rois(rois, coord)


def predict_big_folder(
    model: StarDist2D,
    images_folder: Path,
    results_folder: Path,
    settings: BigImageSettings,
    hooks: predict.Hooks | None = None,
) -> list[str]:
    """Predict every very large image (or stack of them) of a folder block by block (cell 43).

    Outputs as 1.20.8: ``<stem>.tif`` and ``<stem>labels_roi.zip``; for stacks ``<stem>/frame_XXXXX.tif`` and
    ``<stem>_labels_roi/frame_XXXXX.zip``. The zarr copies go to a temporary folder that is removed afterwards.
    """
    hooks = hooks or predict.Hooks()
    results_folder.mkdir(parents=True, exist_ok=True)
    done = []
    with tempfile.TemporaryDirectory(prefix="stardist2d_big_") as workdir:
        for path in data.image_files(images_folder, "large images"):
            hooks.check_cancel()
            hooks.progress(0.0, f"predicting {path.name}")
            image = data.read_image(path)
            frames = image if settings.is_stack else image[np.newaxis]
            for index, frame in enumerate(frames):
                hooks.check_cancel()
                labels, details = predict.predict_big(
                    model, frame, settings.block_size, settings.min_overlap, settings.n_tiles, Path(workdir)
                )
                if settings.is_stack:
                    mask = results_folder / path.stem / _frame_name(index)
                    rois = results_folder / f"{path.stem}_labels_roi" / _frame_name(index).replace(".tif", ".zip")
                else:
                    mask, rois = results_folder / f"{path.stem}.tif", results_folder / f"{path.stem}labels_roi.zip"
                _write_big_outputs(labels, details["coord"], mask, rois, settings)
            done.append(path.name)
    return done
