"""Training settings and the rules of notebook sections 3.2, 3.3 and 4.1 (ZeroCostDL4Mic 1.20.8, cells 20, 23, 28).

Pure: no file access. The values that need the training images (image size, number of images)
are passed in by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

DEFAULT_BATCH_SIZE = 2
DEFAULT_N_RAYS = 32
DEFAULT_PERCENTAGE_VALIDATION = 10
DEFAULT_GRID = 2
DEFAULT_LEARNING_RATE = 0.0003
GRID_CHOICES = (1, 2, 4, 8, 16, 32)
MAX_PATCH_SIZE = 2048
PATCH_MULTIPLE = 16
"""StarDist's U-Net needs patch sides divisible by 16 (four poolings)."""


@dataclass(frozen=True)
class TrainingSettings:
    """What section 3.2 of the notebook asks for (names and defaults as in 1.20.8).

    Attributes
    ----------
    number_of_epochs
        Number of training epochs.
    use_default_advanced_parameters
        Use the defaults below for batch size, rays, validation, grid and learning rate, a patch
        as large as the images and the default number of steps.
    batch_size, number_of_steps, patch_size, percentage_validation, n_rays, grid_parameter, initial_learning_rate
        The advanced parameters, used when ``use_default_advanced_parameters`` is false.
        ``number_of_steps = 0`` also selects the default number of steps.
    """

    number_of_epochs: int = 100
    use_default_advanced_parameters: bool = True
    batch_size: int = DEFAULT_BATCH_SIZE
    number_of_steps: int = 50
    patch_size: int = 512
    percentage_validation: float = DEFAULT_PERCENTAGE_VALIDATION
    n_rays: int = DEFAULT_N_RAYS
    grid_parameter: int = DEFAULT_GRID
    initial_learning_rate: float = DEFAULT_LEARNING_RATE

    def __post_init__(self) -> None:
        _require(self.number_of_epochs >= 1, "number_of_epochs must be at least 1")
        _require(self.batch_size >= 1, "batch_size must be at least 1")
        _require(self.number_of_steps >= 0, "number_of_steps must be 0 (automatic) or more")
        _require(0 < self.percentage_validation < 100, "percentage_validation must be between 0 and 100 (exclusive)")
        _require(self.n_rays >= 3, "n_rays must be at least 3")
        _require(self.grid_parameter in GRID_CHOICES, f"grid_parameter must be one of {GRID_CHOICES}")
        _require(self.initial_learning_rate > 0, "initial_learning_rate must be positive")


@dataclass(frozen=True)
class AugmentationSettings:
    """Section 3.3: data augmentation on or off, and the factor that scales the default number of steps."""

    use_data_augmentation: bool = True
    multiply_dataset_by: int = 4

    def __post_init__(self) -> None:
        _require(1 <= self.multiply_dataset_by <= 10, "multiply_dataset_by must be between 1 and 10")


@dataclass(frozen=True)
class PatchDecision:
    """The patch size used for training and the notes the notebook prints about it."""

    patch_size: int
    notes: tuple[str, ...]


def _require(condition: bool, message: str) -> None:
    """Raise ``ValueError(message)`` when *condition* is false."""
    if not condition:
        raise ValueError(message)


def with_defaults(settings: TrainingSettings) -> TrainingSettings:
    """Return *settings* with the default advanced parameters applied when they are selected (cell 20)."""
    if not settings.use_default_advanced_parameters:
        return settings
    return replace(
        settings,
        batch_size=DEFAULT_BATCH_SIZE,
        n_rays=DEFAULT_N_RAYS,
        percentage_validation=DEFAULT_PERCENTAGE_VALIDATION,
        grid_parameter=DEFAULT_GRID,
        initial_learning_rate=DEFAULT_LEARNING_RATE,
    )


def decide_patch_size(settings: TrainingSettings, image_shape: tuple[int, ...]) -> PatchDecision:
    """Return the training patch size for images of *image_shape* (YX first), with the cell 20 failsafes.

    Difference from 1.20.8: a size that is not a multiple of 16 is rounded *down to the nearest*
    multiple of 16 (1.20.8 removed a further 16 pixels, e.g. 500 gave 480 instead of 496).

    Raises
    ------
    ValueError
        If the images are smaller than 16 pixels in Y or X.
    """
    smallest_side = min(image_shape[0], image_shape[1])
    _require(smallest_side >= PATCH_MULTIPLE, f"images must be at least {PATCH_MULTIPLE} pixels in Y and X")
    patch = smallest_side if settings.use_default_advanced_parameters else settings.patch_size
    notes: list[str] = []
    if patch > smallest_side:
        patch = smallest_side
        notes.append(f"Your chosen patch_size is bigger than the xy dimension of your image; it is now {patch}")
    if patch > MAX_PATCH_SIZE:
        patch = MAX_PATCH_SIZE
        notes.append(f"Your image dimension is large; the patch_size is now {patch}")
    if patch % PATCH_MULTIPLE:
        patch = (patch // PATCH_MULTIPLE) * PATCH_MULTIPLE
        notes.append(f"Your chosen patch_size is not divisible by {PATCH_MULTIPLE}; it is now {patch}")
    return PatchDecision(patch, tuple(notes))


def number_of_steps(
    settings: TrainingSettings,
    augmentation: AugmentationSettings,
    image_shape: tuple[int, ...],
    patch_size: int,
    n_images: int,
) -> int:
    """Return the steps per epoch (cell 28): the setting, or the default formula when selected.

    The default is ``int(Y*X / patch²) * (int(n_images / batch_size) + 1)``, multiplied by the
    augmentation factor when augmentation is on.
    """
    if not (settings.use_default_advanced_parameters or settings.number_of_steps == 0):
        return settings.number_of_steps
    patches_per_image = int(image_shape[0] * image_shape[1] / (patch_size * patch_size))
    steps = patches_per_image * (int(n_images / settings.batch_size) + 1)
    if augmentation.use_data_augmentation:
        steps *= augmentation.multiply_dataset_by
    return steps


def validation_count(percentage_validation: float, n_images: int) -> int:
    """Return how many images go to validation (cell 28: at least one).

    Raises
    ------
    ValueError
        If there are fewer than two images (one is needed for training and one for validation).
    """
    _require(n_images > 1, "not enough training data: at least two image/mask pairs are needed")
    return max(1, int(round(percentage_validation / 100 * n_images)))


def config2d_arguments(settings: TrainingSettings, n_channel: int, patch_size: int) -> dict[str, Any]:
    """Return the keyword arguments of ``stardist.models.Config2D`` used by the notebook (cell 28)."""
    return {
        "n_rays": settings.n_rays,
        "use_gpu": False,
        "train_batch_size": settings.batch_size,
        "n_channel_in": n_channel,
        "train_patch_size": (patch_size, patch_size),
        "grid": (settings.grid_parameter, settings.grid_parameter),
        "train_learning_rate": settings.initial_learning_rate,
    }
