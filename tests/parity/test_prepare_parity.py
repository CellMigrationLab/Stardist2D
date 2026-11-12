"""Training data preparation equals cell 28 of 1.20.8 (normalisation, hole filling, split)."""

import numpy as np
import pytest
from csbdeep.utils import normalize
from stardist import fill_label_holes

from reference_notebook import run
from stardist2d import prepare
from stardist2d.config import validation_count

STARTS = (
    "X = [normalize",
    "Y = [fill_label_holes",
    "rng =",
    "ind =",
    "n_val =",
    "ind_train, ind_val",
    "X_val, Y_val",
    "X_trn, Y_trn",
    "if n_channel == 4",
)


def _data(n: int, channels: int):
    rng = np.random.default_rng(5)
    shape = (40, 48) if channels == 1 else (40, 48, channels)
    images = [rng.integers(0, 4000, shape).astype(np.uint16) for _ in range(n)]
    labels = []
    for _ in range(n):
        label = np.zeros((40, 48), np.uint16)
        label[5:15, 5:15] = 1
        label[8:11, 8:11] = 0  # a hole fill_label_holes must close
        label[20:30, 25:40] = 2
        labels.append(label)
    return images, labels


@pytest.mark.parametrize(("n", "channels", "percentage"), [(5, 1, 10), (9, 1, 25), (4, 1, 50)])
def test_preparation_and_split_match_1_20_8(n, channels, percentage):
    images, labels = _data(n, channels)
    namespace = {
        "X": list(images),
        "Y": list(labels),
        "normalize": normalize,
        "fill_label_holes": fill_label_holes,
        "tqdm": lambda items: items,
        "np": np,
        "percentage": percentage / 100,
        "axis_norm": (0, 1),
        "n_channel": channels,
        "print": lambda *args, **kwargs: None,
    }
    run(28, lambda code: code.startswith(STARTS), namespace)

    x = prepare.drop_alpha(images)
    x = [prepare.normalize_image(image, prepare.training_axes(image)) for image in x]
    y = [prepare.fill_holes(label) for label in labels]
    split = prepare.split_training_data(x, y, validation_count(percentage, n))
    for ours, theirs in [
        (split.x_train, "X_trn"),
        (split.y_train, "Y_trn"),
        (split.x_val, "X_val"),
        (split.y_val, "Y_val"),
    ]:
        assert len(ours) == len(namespace[theirs])
        for a, b in zip(ours, namespace[theirs], strict=True):
            np.testing.assert_array_equal(a, b)


def test_channels_are_normalised_together_for_training_and_prediction():
    # Intentional difference from 1.20.8, which trained with channels normalised separately (axis (0, 1)).
    for image in (np.zeros((4, 4, 3)), np.zeros((4, 4, 4))[..., :3]):
        assert prepare.training_axes(image) == prepare.prediction_axes(image) == (0, 1, 2)
    assert prepare.training_axes(np.zeros((4, 4))) == (0, 1)


@pytest.mark.parametrize("channels", [3, 4])
def test_rgb_training_preparation_differs_from_1_20_8_only_by_the_joint_normalisation(channels):
    images, labels = _data(3, channels)
    namespace = {
        "X": list(images),
        "Y": list(labels),
        "normalize": normalize,
        "fill_label_holes": fill_label_holes,
        "tqdm": lambda items: items,
        "np": np,
        "percentage": 0.34,
        "axis_norm": (0, 1, 2),
        "n_channel": channels,
        "print": lambda *args, **kwargs: None,
    }
    run(28, lambda code: code.startswith(STARTS) and not code.startswith("axis_norm"), namespace)
    x = [prepare.normalize_image(image, prepare.training_axes(image)) for image in prepare.drop_alpha(images)]
    split = prepare.split_training_data(x, labels, 1)
    for ours, theirs in zip(split.x_train, namespace["X_trn"], strict=True):
        np.testing.assert_array_equal(ours, theirs)


def test_split_refuses_mismatched_or_too_small_inputs():
    with pytest.raises(ValueError, match="label images"):
        prepare.split_training_data([np.zeros(1)] * 3, [np.zeros(1)] * 2, 1)
    with pytest.raises(ValueError, match="for validation"):
        prepare.split_training_data([np.zeros(1)] * 2, [np.zeros(1)] * 2, 2)
