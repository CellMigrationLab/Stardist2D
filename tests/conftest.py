"""Shared fixtures: synthetic nuclei images and StarDist's pretrained model in a temporary model folder."""

import shutil

import numpy as np
import pytest
import tifffile


def synthetic_pair(seed: int, shape: tuple[int, int] = (128, 128), n: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Return a noisy image of bright discs and its label image."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    labels = np.zeros(shape, np.uint16)
    for label in range(1, n + 1):
        cy, cx, r = rng.integers(10, shape[0] - 10), rng.integers(10, shape[1] - 10), rng.integers(4, 8)
        labels[((yy - cy) ** 2 + (xx - cx) ** 2 <= r * r) & (labels == 0)] = label
    image = (labels > 0) * 1000.0 + rng.normal(100, 30, shape)
    return image.astype(np.float32), labels


@pytest.fixture
def training_folders(tmp_path):
    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    for index in range(4):
        image, labels = synthetic_pair(index)
        tifffile.imwrite(images / f"cell_{index}.tif", image)
        tifffile.imwrite(masks / f"cell_{index}.tif", labels)
    return images, masks


@pytest.fixture(scope="session")
def pretrained_folder(tmp_path_factory):
    from stardist.models import StarDist2D

    target = tmp_path_factory.mktemp("models") / "versatile"
    source = StarDist2D.from_pretrained("2D_versatile_fluo").logdir
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("*.zip"))
    return target
