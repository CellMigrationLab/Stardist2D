"""Find and read the images of a run (1.20.8, cells 17, 20, 28, 36, 41).

1.20.8 reads ``*.tif`` files only, in sorted order, and pairs images with masks by file name.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np

TEST_DATASET_URL = "https://zenodo.org/record/7949940/files/Stardist.zip?download=1"
"""The test dataset of section 3.1 (Zenodo record 7949940)."""
IMAGE_PATTERN = "*.tif"


def _existing_folder(folder: Path, role: str) -> Path:
    """Return *folder* when it is an existing folder; refuse empty or missing paths with a clear message."""
    if not str(folder).strip() or str(folder) == ".":
        raise ValueError(f"no {role} folder was given")
    if not folder.is_dir():
        raise FileNotFoundError(f"the {role} folder does not exist: {folder}")
    return folder


def image_files(folder: Path, role: str = "image") -> list[Path]:
    """Return the ``*.tif`` files of *folder*, sorted by name.

    Raises
    ------
    FileNotFoundError
        If the folder does not exist or holds no ``.tif`` file.
    """
    files = sorted(_existing_folder(folder, role).glob(IMAGE_PATTERN))
    if not files:
        raise FileNotFoundError(f"no .tif file in the {role} folder {folder}")
    return files


def training_pairs(images_folder: Path, masks_folder: Path) -> list[tuple[Path, Path]]:
    """Return ``(image, mask)`` file pairs matched by name.

    Difference from 1.20.8: every image needs a mask of the same name and every mask an image. 1.20.8
    compared the two sorted lists with ``zip``, which silently ignored extra files in the longer list.

    Raises
    ------
    ValueError
        If the two folders do not hold the same file names.
    """
    images = {path.name: path for path in image_files(images_folder, "training images")}
    masks = {path.name: path for path in image_files(masks_folder, "training masks")}
    if images.keys() != masks.keys():
        only_images = sorted(images.keys() - masks.keys())
        only_masks = sorted(masks.keys() - images.keys())
        raise ValueError(
            "images and masks must have the same file names; "
            f"without a mask: {only_images or 'none'}; without an image: {only_masks or 'none'}"
        )
    return [(images[name], masks[name]) for name in sorted(images)]


def read_image(path: Path) -> np.ndarray:
    """Return the pixels of a TIFF file."""
    import tifffile

    return tifffile.imread(path)


def read_images(paths: list[Path]) -> list[np.ndarray]:
    """Return the pixels of every file of *paths*, in order."""
    return [read_image(path) for path in paths]


def download_test_dataset(destination: Path, url: str = TEST_DATASET_URL, timeout: float = 600) -> Path:
    """Download and unpack the section 3.1 test dataset into *destination* (``destination/Stardist/...``).

    Raises
    ------
    RuntimeError
        If the download fails.
    """
    import requests

    response = requests.get(url, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(f"the test dataset could not be downloaded from {url} (HTTP {response.status_code})")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        archive.extractall(destination)
    return destination
