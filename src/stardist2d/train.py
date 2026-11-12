"""Train a StarDist 2D model (1.20.8, sections 4.1 and 4.2, cells 28 and 30)."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from stardist2d import augment, config, data, export, model, prepare
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


def _keras_callback(
    on_epoch: Callable[[int, dict[str, float]], None] | None, on_batch: Callable[[], None] | None
) -> tf_keras.callbacks.Callback:
    """Return a Keras callback calling ``on_epoch(epoch, logs)`` after each epoch and ``on_batch()`` after each batch.

    An exception raised by either stops training at once (Keras lets it through): the tools raise ``Cancelled`` from
    ``on_batch`` so that Cancel stops within a batch. What StarDist saved until then (``weights_best.h5``, written at
    every epoch whose validation loss improved) stays in the model folder.
    """
    check_keras_backend()
    import tf_keras

    class _ProgressCallback(tf_keras.callbacks.Callback):
        def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
            if on_epoch:
                on_epoch(epoch, dict(logs or {}))

        def on_train_batch_end(self, batch: int, logs: dict[str, float] | None = None) -> None:
            if on_batch:
                on_batch()

    return _ProgressCallback()


def train(
    star_dist: StarDist2D,
    training: TrainingData,
    epochs: int,
    use_augmentation: bool,
    on_epoch: Callable[[int, dict[str, float]], None] | None = None,
    on_batch: Callable[[], None] | None = None,
) -> TrainingResult:
    """Train *star_dist* and optimise its thresholds on the validation images (cell 30).

    *on_epoch* is called after every epoch with its logs; *on_batch* after every batch. Either may raise to stop
    training (see :func:`_keras_callback`); then no thresholds are optimised and nothing is returned.
    """
    split = training.split
    if on_epoch or on_batch:
        # StarDist's train() has no callbacks argument: it uses model.callbacks, made by prepare_for_training().
        star_dist.prepare_for_training()
        star_dist.callbacks.append(_keras_callback(on_epoch, on_batch))
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
    rows = [
        [loss, history["val_loss"][index], rates[index] if rates else None]
        for index, loss in enumerate(history["loss"])
    ]
    # Written atomically: the tools rewrite it after every epoch, and a cancelled run is stopped mid-way.
    return export.write_csv(
        model_folder / "Quality Control" / "training_evaluation.csv", [["loss", "val_loss", "learning rate"], *rows]
    )


def new_model(training: TrainingData, model_folder: Path, overwrite: bool) -> StarDist2D:
    """Create the model of section 4.1 for *training* in *model_folder*."""
    arguments = config.config2d_arguments(training.settings, training.n_channel, training.patch_size)
    return model.create_model(arguments, model_folder, overwrite=overwrite)


def elapsed(seconds: float) -> tuple[int, int, int]:
    """Return *seconds* as (hours, minutes, seconds), as 1.20.8 prints the training time."""
    minutes, rest = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return int(hours), int(minutes), int(np.round(rest))
