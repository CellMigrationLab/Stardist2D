"""config.py follows the rules of cells 20 and 28 of 1.20.8 (one intentional fix: the patch-size rounding)."""

import pytest

from reference_notebook import run
from stardist2d.config import (
    AugmentationSettings,
    TrainingSettings,
    config2d_arguments,
    decide_patch_size,
    number_of_steps,
    validation_count,
    with_defaults,
)


class _Colors:
    WARNING = ENDC = ""


def reference_patch_size(use_defaults: bool, patch_size: int, shape: tuple[int, int]) -> int:
    """Run the patch-size statements of cell 20."""
    namespace = {
        "Use_Default_Advanced_Parameters": use_defaults,
        "patch_size": patch_size,
        "Image_Y": shape[0],
        "Image_X": shape[1],
        "bcolors": _Colors,
        "print": lambda *args, **kwargs: None,
    }
    run(20, lambda code: "patch_size" in code and code.startswith("if"), namespace)
    return namespace["patch_size"]


@pytest.mark.parametrize(
    ("use_defaults", "patch", "shape"),
    [
        (True, 512, (256, 256)),
        (True, 512, (3000, 4000)),
        (False, 512, (600, 700)),
        (False, 256, (1024, 1024)),
        (False, 128, (100, 300)),
        (True, 0, (64, 80)),
    ],
)
def test_patch_sizes_that_are_multiples_of_16_match_1_20_8(use_defaults, patch, shape):
    expected = reference_patch_size(use_defaults, patch, shape)
    settings = TrainingSettings(use_default_advanced_parameters=use_defaults, patch_size=patch)
    if expected % 16 == 0 and min(shape) % 16 == 0 or patch % 16 == 0 and patch <= min(shape):
        assert decide_patch_size(settings, shape).patch_size == expected


def test_intentional_fix_patch_size_rounds_down_to_the_nearest_multiple_of_16():
    # 1.20.8 gave 480 for 500 (it removed one more multiple of 16); now 496. Listed in CHANGELOG.md.
    assert reference_patch_size(False, 500, (1000, 1000)) == 480
    decision = decide_patch_size(TrainingSettings(use_default_advanced_parameters=False, patch_size=500), (1000, 1000))
    assert decision.patch_size == 496
    assert "not divisible by 16" in decision.notes[0]


def test_images_smaller_than_16_pixels_are_refused():
    with pytest.raises(ValueError, match="at least 16 pixels"):
        decide_patch_size(TrainingSettings(), (8, 100))


def reference_steps(use_defaults, steps, augmentation, factor, shape, patch, n_images, batch):
    namespace = {
        "Use_Default_Advanced_Parameters": use_defaults,
        "number_of_steps": steps,
        "Image_X": shape[1],
        "Image_Y": shape[0],
        "patch_size": patch,
        "X": [None] * n_images,
        "batch_size": batch,
        "Use_Data_augmentation": augmentation,
        "Multiply_dataset_by": factor,
    }
    run(28, lambda code: code.startswith("if (Use_Default_Advanced_Parameters) or"), namespace)
    return namespace["number_of_steps"]


@pytest.mark.parametrize(
    ("use_defaults", "steps", "augmentation", "factor", "shape", "patch", "n_images", "batch"),
    [
        (True, 50, True, 4, (512, 512), 512, 10, 2),
        (True, 50, False, 4, (1024, 768), 256, 7, 2),
        (False, 0, True, 3, (512, 512), 128, 5, 4),
        (False, 30, True, 4, (512, 512), 128, 5, 4),
    ],
)
def test_number_of_steps_matches_1_20_8(use_defaults, steps, augmentation, factor, shape, patch, n_images, batch):
    expected = reference_steps(use_defaults, steps, augmentation, factor, shape, patch, n_images, batch)
    settings = TrainingSettings(use_default_advanced_parameters=use_defaults, number_of_steps=steps, batch_size=batch)
    augmentation_settings = AugmentationSettings(use_data_augmentation=augmentation, multiply_dataset_by=factor)
    assert number_of_steps(settings, augmentation_settings, shape, patch, n_images) == expected


@pytest.mark.parametrize(("percentage", "n_images"), [(10, 2), (10, 30), (25, 9), (50, 3)])
def test_validation_count_matches_1_20_8(percentage, n_images):
    namespace = {"percentage": percentage / 100, "ind": list(range(n_images))}
    run(28, lambda code: code.startswith("n_val ="), namespace)
    assert validation_count(percentage, n_images) == namespace["n_val"]


def test_one_image_is_refused():
    with pytest.raises(ValueError, match="at least two"):
        validation_count(10, 1)


def test_defaults_replace_the_advanced_parameters():
    custom = TrainingSettings(batch_size=8, n_rays=64, grid_parameter=4, initial_learning_rate=0.01)
    assert with_defaults(custom).batch_size == 2
    assert with_defaults(custom).n_rays == 32
    assert with_defaults(TrainingSettings(use_default_advanced_parameters=False, batch_size=8)).batch_size == 8


def test_config2d_arguments_are_those_of_cell_28():
    args = config2d_arguments(TrainingSettings(), n_channel=3, patch_size=256)
    assert args == {
        "n_rays": 32,
        "use_gpu": False,
        "train_batch_size": 2,
        "n_channel_in": 3,
        "train_patch_size": (256, 256),
        "grid": (2, 2),
        "train_learning_rate": 0.0003,
    }


def test_invalid_settings_are_refused_with_a_reason():
    with pytest.raises(ValueError, match="grid_parameter"):
        TrainingSettings(grid_parameter=3)
    with pytest.raises(ValueError, match="multiply_dataset_by"):
        AugmentationSettings(multiply_dataset_by=11)
