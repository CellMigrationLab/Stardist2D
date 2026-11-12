"""Train a StarDist 2D model (1.20.8, sections 4.1 and 4.2, cells 28 and 30)."""

from __future__ import annotations

import csv
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from stardist2d import augment, config, data, model, prepare
from stardist2d.keras_backend import check_keras_backend

if TYPE_CHECKING:
    import tf_keras
    from stardist.models import StarDist2D


@dataclass(frozen=True)
class TrainingData:
    """Everything section 4.1 prepares before training starts."""

    split: prepare.TrainingSplit
    n_channel: int
    image_shape: tuple[int, ...]
    patch_size: int
    patch_notes: tuple[str, ...]
    steps_per_epoch: int
    settings: config.TrainingSettings


@dataclass(frozen=True)
class TrainingResult:
    """The trained model, its loss history and how long training took (seconds)."""

    model: StarDist2D
    history: dict[str, list[float]]
    seconds: float


def load_training_data(
    images_folder: Path,
    masks_folder: Path,
    settings: config.TrainingSettings,
    augmentation: config.AugmentationSettings,
) -> TrainingData:
    """Read, normalise and split the training pairs and work out patch size and steps (cells 20 and 28).

    Difference from 1.20.8: the image size used for the patch size and the default steps is that of the
    first image in name order (1.20.8 used a randomly chosen image).
    """
    settings = config.with_defaults(settings)
    pairs = data.training_pairs(images_folder, masks_folder)
    images = prepare.drop_alpha(data.read_images([image for image, _ in pairs]))
    labels = data.read_images([mask for _, mask in pairs])
    image_shape = tuple(images[0].shape)
    patch = config.decide_patch_size(settings, image_shape)
    images = [prepare.normalize_image(image, prepare.training_axes(image)) for image in images]
    labels = [prepare.fill_holes(label) for label in labels]
    split = prepare.split_training_data(
        images, labels, config.validation_count(settings.percentage_validation, len(images))
    )
    steps = config.number_of_steps(settings, augmentation, image_shape, patch.patch_size, len(images))
    n_channel = prepare.channel_count(images[0])
    return TrainingData(split, n_channel, image_shape, patch.patch_size, patch.notes, steps, settings)


def _keras_callback(on_epoch: Callable[[int, dict[str, float]], None]) -> tf_keras.callbacks.Callback:
    """Return a Keras callback that calls ``on_epoch(epoch, logs)`` after every epoch."""
    check_keras_backend()
    import tf_keras

    class _EpochCallback(tf_keras.callbacks.Callback):
        def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
            on_epoch(epoch, dict(logs or {}))

    return _EpochCallback()


def train(
    star_dist: StarDist2D,
    training: TrainingData,
    epochs: int,
    use_augmentation: bool,
    on_epoch: Callable[[int, dict[str, float]], None] | None = None,
) -> TrainingResult:
    """Train *star_dist* and optimise its thresholds on the validation images (cell 30)."""
    split = training.split
    if on_epoch:
        # StarDist's train() has no callbacks argument: it uses model.callbacks, made by prepare_for_training().
        star_dist.prepare_for_training()
        star_dist.callbacks.append(_keras_callback(on_epoch))
    start = time.time()
    history = star_dist.train(
        split.x_train,
        split.y_train,
        validation_data=(split.x_val, split.y_val),
        augmenter=augment.augmenter if use_augmentation else None,
        epochs=epochs,
        steps_per_epoch=training.steps_per_epoch,
    )
    star_dist.optimize_thresholds(split.x_val, split.y_val)
    return TrainingResult(
        star_dist, {key: list(map(float, value)) for key, value in history.history.items()}, time.time() - start
    )


def write_training_evaluation(model_folder: Path, history: dict[str, list[float]]) -> Path:
    """Write ``Quality Control/training_evaluation.csv`` with 1.20.8's columns (loss, val_loss, learning rate)."""
    rates = history.get("learning_rate", history.get("lr"))
    path = model_folder / "Quality Control" / "training_evaluation.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["loss", "val_loss", "learning rate"])
        for index, loss in enumerate(history["loss"]):
            writer.writerow([loss, history["val_loss"][index], rates[index] if rates else None])
    return path


def new_model(training: TrainingData, model_folder: Path, overwrite: bool) -> StarDist2D:
    """Create the model of section 4.1 for *training* in *model_folder*."""
    arguments = config.config2d_arguments(training.settings, training.n_channel, training.patch_size)
    return model.create_model(arguments, model_folder, overwrite=overwrite)


def elapsed(seconds: float) -> tuple[int, int, int]:
    """Return *seconds* as (hours, minutes, seconds), as 1.20.8 prints the training time."""
    minutes, rest = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return int(hours), int(minutes), int(np.round(rest))
