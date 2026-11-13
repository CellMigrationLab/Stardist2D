"""Create and load StarDist 2D models (1.20.8, cells 28, 36, 41, 43).

TensorFlow is imported only inside the functions, after :func:`stardist2d.keras_backend.check_keras_backend`.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from stardist2d import model_catalogue
from stardist2d.keras_backend import check_keras_backend

if TYPE_CHECKING:
    from stardist.models import StarDist2D

PRETRAINED_MODELS = ("2D_versatile_fluo", "2D_versatile_he", "2D_paper_dsb2018")
"""StarDist's registered 2D models, downloaded by ``StarDist2D.from_pretrained`` into ``~/.keras``."""
CUSTOM_MODEL = "Custom model folder"
_REGISTERED: dict[str, str] = {}
"""Catalogue model name -> the address it is registered with in StarDist's registry (this process)."""


@dataclass(frozen=True)
class ModelChoice:
    """Which model to use: a pretrained StarDist model name, or :data:`CUSTOM_MODEL` with its folder."""

    source: str
    folder: Path | None = None

    def __post_init__(self) -> None:
        if self.source == CUSTOM_MODEL:
            if self.folder is None or not str(self.folder).strip():
                raise ValueError("choose the folder of your StarDist 2D model (it holds config.json)")
        elif self.source not in PRETRAINED_MODELS and model_catalogue.by_label(self.source) is None:
            choices = (*PRETRAINED_MODELS, *(m.label for m in model_catalogue.CATALOGUE), CUSTOM_MODEL)
            raise ValueError(f"unknown model {self.source!r}: choose one of {choices}")


def _stardist2d() -> type[StarDist2D]:
    """Return the ``StarDist2D`` class, imported after Keras 2 is selected, with the catalogue models registered."""
    check_keras_backend()
    from csbdeep.models.pretrained import register_model
    from stardist.models import StarDist2D

    for entry in model_catalogue.CATALOGUE:
        url = model_catalogue.base_url() + entry.name + ".zip"
        if _REGISTERED.get(entry.name) != url:  # once per address: CSBDeep warns at every re-registration
            register_model(StarDist2D, entry.name, url, entry.sha256)
            _REGISTERED[entry.name] = url
    return StarDist2D


def load_model_folder(folder: Path) -> StarDist2D:
    """Load a trained StarDist 2D model from its folder (``config.json``, weights, ``thresholds.json``).

    Raises
    ------
    FileNotFoundError
        If the folder or its ``config.json`` does not exist.
    """
    folder = Path(folder)
    if not (folder / "config.json").is_file():
        raise FileNotFoundError(f"{folder} is not a StarDist model folder: config.json is missing")
    return _stardist2d()(None, name=folder.name, basedir=str(folder.parent))


def from_registry(name: str) -> StarDist2D:
    """Return a registered pretrained model (StarDist's or the catalogue's), downloaded once and checked (SHA-256).

    Raises
    ------
    RuntimeError
        If it cannot be downloaded or fails its checksum: CSBDeep's ``from_pretrained`` then prints a table and returns
        None, which used to surface later as "'NoneType' object has no attribute ...".
    """
    star_dist = _stardist2d().from_pretrained(name)
    if star_dist is None:
        raise RuntimeError(
            f"the pretrained model {name} could not be downloaded, or the download failed its checksum: check the "
            "internet connection and try again, or choose a model folder on this computer"
        )
    return star_dist


def load_model(choice: ModelChoice) -> StarDist2D:
    """Load the model of *choice*: a pretrained one (downloaded once) or a model folder."""
    if choice.source == CUSTOM_MODEL and choice.folder is not None:
        return load_model_folder(choice.folder)
    entry = model_catalogue.by_label(choice.source)
    return from_registry(entry.name if entry else choice.source)


def create_model(config_arguments: dict[str, Any], model_folder: Path, overwrite: bool = False) -> StarDist2D:
    """Create a new, untrained model in *model_folder* (cell 28).

    1.20.8 deletes an existing folder of the same name without asking. Here it is deleted only with
    ``overwrite=True``; the notebook asks first.

    Raises
    ------
    FileExistsError
        If *model_folder* exists and *overwrite* is false.
    """
    model_folder = Path(model_folder)
    if model_folder.exists():
        if not overwrite:
            raise FileExistsError(f"{model_folder} already exists: choose another model name or allow overwriting it")
        shutil.rmtree(model_folder)
    star_dist = _stardist2d()
    from stardist.models import Config2D

    return star_dist(Config2D(**config_arguments), name=model_folder.name, basedir=str(model_folder.parent))


def load_weights(model: StarDist2D, weights_file: Path) -> None:
    """Load pretrained weights into *model* (cell 28).

    Raises
    ------
    ValueError
        If the weights do not fit the network (different rays, grid or channels), with what to check.
    """
    try:
        model.load_weights(str(weights_file))
    except (ValueError, OSError) as error:
        raise ValueError(
            f"the pretrained weights {weights_file} could not be loaded: the network configuration differs "
            "(check n_rays, grid_parameter and the number of channels in section 3.2)"
        ) from error


def object_larger_than_field_of_view(model: StarDist2D, labels: list[np.ndarray]) -> bool:
    """Return whether the median object is larger than the network's field of view (cell 28 warning)."""
    from stardist import calculate_extents

    median_size = calculate_extents(list(labels), np.median)
    field_of_view = np.array(model._axes_tile_overlap("YX"))
    return bool(np.any(median_size > field_of_view))
