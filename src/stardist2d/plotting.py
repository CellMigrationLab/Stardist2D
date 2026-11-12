"""Figures of the notebook (1.20.8, cells 20, 34, 36 and 41). Every function returns a figure; saving is optional.

UI layer: drawing only. The numbers come from the other modules.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure


def _figure(size: tuple[float, float]) -> Figure:
    """Return a new figure that is not attached to pyplot (so nothing is shown unless the caller displays it)."""
    from matplotlib.figure import Figure

    return Figure(figsize=size)


def _save(figure: Figure, path: Path | None) -> Figure:
    """Save *figure* to *path* when one is given, as 1.20.8 does (tight bounding box, no padding)."""
    if path is not None:
        from stardist2d.export import atomic_write

        atomic_write(path, lambda temporary: figure.savefig(temporary, bbox_inches="tight", pad_inches=0))
    return figure


def _show_image(axis: Axes, image: np.ndarray) -> None:
    """Draw *image* with 1.20.8's display: 99 % contrast and the magma colour map for one channel, RGB otherwise."""
    from astropy.visualization import simple_norm

    if image.ndim == 3:
        axis.imshow(image[..., :3])
    else:
        axis.imshow(image, norm=simple_norm(image, percent=99), cmap="magma", interpolation="nearest")
    axis.axis("off")


def _label_colours() -> Colormap:
    """Return StarDist's random label colour map, seeded as 1.20.8 does (16)."""
    from stardist import random_label_cmap

    state = np.random.get_state()
    np.random.seed(16)
    colours = random_label_cmap()
    np.random.set_state(state)
    return colours


def training_pair(image: np.ndarray, labels: np.ndarray, path: Path | None = None) -> Figure:
    """Return section 3.2's preview of a training image and its mask (``TrainingDataExample_StarDist2D.png``)."""
    figure = _figure((16, 8))
    source, target = figure.subplots(1, 2)
    _show_image(source, image)
    source.set_title("Training source")
    target.imshow(labels, interpolation="nearest", cmap=_label_colours())
    target.set_title("Training target")
    target.axis("off")
    return _save(figure, path)


def loss_curves(loss: Sequence[float], val_loss: Sequence[float], path: Path | None = None) -> Figure:
    """Return section 5.1's training and validation loss curves, linear and log scale (``lossCurvePlots.png``)."""
    figure = _figure((15, 10))
    epochs = range(len(loss))
    for axis, scale in zip(figure.subplots(2, 1), ("linear", "log"), strict=True):
        axis.plot(epochs, loss, label="Training loss")
        axis.plot(epochs, val_loss, label="Validation loss")
        axis.set_yscale(scale)
        axis.set_title(f"Training loss and validation loss vs. epoch number ({scale} scale)")
        axis.set_ylabel("Loss")
        axis.set_xlabel("Epoch number")
        axis.legend()
    return _save(figure, path)


def qc_example(
    image: np.ndarray, truth: np.ndarray, prediction: np.ndarray, iou: float, path: Path | None = None
) -> Figure:
    """Return section 5.2's four panels: input, ground truth, prediction, overlay (``QC_example_data.png``)."""
    figure = _figure((25, 5))
    axes = figure.subplots(1, 4)
    _show_image(axes[0], image)
    axes[0].set_title("Input")
    truth_mask, predicted_mask = (truth > 0) * 255, (prediction > 0) * 255
    for axis, mask, colours, title in (
        (axes[1], truth_mask, "Greens", "Ground Truth"),
        (axes[2], predicted_mask, "Purples", "Prediction"),
    ):
        axis.imshow(mask, aspect="equal", cmap=colours)
        axis.set_title(title)
        axis.axis("off")
    axes[3].imshow(truth_mask, cmap="Greens")
    axes[3].imshow(predicted_mask, alpha=0.5, cmap="Purples")
    axes[3].set_title(f"Ground Truth and Prediction, Intersection over Union:{round(iou, 3)}")
    axes[3].axis("off")
    return _save(figure, path)


def overlay(image: np.ndarray, labels: np.ndarray, path: Path | None = None) -> Figure:
    """Return section 6.1's example: the normalised image with its labels on top (``<name>_overlay.tif``)."""
    figure = _figure((10, 10))
    axis = figure.subplots()
    axis.imshow(image if image.ndim == 2 else image[..., :3], clim=(0, 1), cmap="gray")
    axis.imshow(labels, cmap=_label_colours(), alpha=0.5)
    axis.axis("off")
    return _save(figure, path)


def objects_per_frame(counts: Sequence[tuple[int, int]], path: Path | None = None) -> Figure:
    """Return the stack plot of section 6.1: number of objects against frame number."""
    figure = _figure((20, 5))
    axis = figure.subplots()
    axis.plot([frame for frame, _ in counts], [n for _, n in counts], label="Number of objects")
    axis.set_title("Number of objects vs frame number")
    axis.set_ylabel("Number of detected objects")
    axis.set_xlabel("Frame number")
    axis.legend()
    return _save(figure, path)
