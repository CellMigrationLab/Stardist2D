"""Data augmentation of notebook section 3.3 (ZeroCostDL4Mic 1.20.8, cell 23), unchanged.

Flips and rotations by 90 degrees, a random intensity scale and offset, and a little Gaussian
noise. Like 1.20.8 it draws from NumPy's global random generator, so ``np.random.seed`` makes a
run repeatable.
"""

from __future__ import annotations

import numpy as np


def random_fliprot(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return *img* and *mask* with the same random axis permutation and flips.

    Parameters
    ----------
    img
        Image with at least as many dimensions as *mask* (extra trailing axes are channels).
    mask
        Label image.
    """
    if img.ndim < mask.ndim:
        raise ValueError("the image must have at least as many dimensions as the mask")
    axes = tuple(range(mask.ndim))
    perm = tuple(np.random.permutation(axes))
    img = img.transpose(perm + tuple(range(mask.ndim, img.ndim)))
    mask = mask.transpose(perm)
    for ax in axes:
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask


def random_intensity_change(img: np.ndarray) -> np.ndarray:
    """Return *img* scaled by a random factor in [0.6, 2) and shifted by a random offset in [-0.2, 0.2)."""
    return img * np.random.uniform(0.6, 2) + np.random.uniform(-0.2, 0.2)


def augmenter(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return an augmented copy of one image/label pair (the ``augmenter`` StarDist's training calls)."""
    x, y = random_fliprot(x, y)
    x = random_intensity_change(x)
    sigma = 0.02 * np.random.uniform(0, 1)
    x = x + sigma * np.random.normal(0, 1, x.shape)
    return x, y
