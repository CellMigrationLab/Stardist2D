"""augment.py does exactly what cell 23 of 1.20.8 does, draw for draw."""

import numpy as np
import pytest

from reference_notebook import functions
from stardist2d import augment

REFERENCE = functions(23, ("random_fliprot", "random_intensity_change", "augmenter"), {"np": np})


@pytest.mark.parametrize("shape", [(32, 48), (32, 48, 3)])
def test_augmenter_matches_1_20_8_with_the_same_seed(shape):
    rng = np.random.default_rng(0)
    image = rng.random(shape)
    mask = rng.integers(0, 5, shape[:2])
    for seed in range(5):
        np.random.seed(seed)
        expected = REFERENCE["augmenter"](image, mask)
        np.random.seed(seed)
        got = augment.augmenter(image, mask)
        np.testing.assert_array_equal(got[0], expected[0])
        np.testing.assert_array_equal(got[1], expected[1])


def test_a_mask_with_more_dimensions_than_the_image_is_refused():
    with pytest.raises(ValueError, match="at least as many dimensions"):
        augment.random_fliprot(np.zeros((4, 4)), np.zeros((4, 4, 2)))
