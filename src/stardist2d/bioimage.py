"""Export a trained model to the BioImage Model Zoo (1.20.8, section 5.3, cell 38)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from stardist.models import StarDist2D

REFERENCES = (
    ("Uwe Schmidt et al. MICCAI 2018", "https://doi.org/10.1007/978-3-030-00934-2_30"),
    ("Martin Weigert et al. WACV 2020", "https://doi.org/10.1109/WACV45572.2020.9093435"),
    ("Lucas von Chamier et al. biorXiv 2020", "https://doi.org/10.1101/2020.03.20.000133"),
)
"""The references 1.20.8 lists in section 5.3."""


@dataclass(frozen=True)
class ModelMetadata:
    """What section 5.3 asks about the model (names of 1.20.8)."""

    name: str
    authors: tuple[str, ...]
    affiliations: tuple[str, ...] = ()
    description: str = ""
    license: str = "MIT"
    references: tuple[tuple[str, str], ...] = field(default=REFERENCES)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("give the trained model a name")
        if not self.authors:
            raise ValueError("give at least one author")


def parse_list(text: str) -> tuple[str, ...]:
    """Return the items of 1.20.8's ``"[Author 1, Author 2]"`` text fields."""
    return tuple(item.strip() for item in text.strip().strip("[]").split(",") if item.strip())


def spec_overrides(metadata: ModelMetadata) -> dict[str, object]:
    """Return the BioImage.IO fields filled from *metadata*.

    Difference from 1.20.8: 1.20.8 asked for these fields but never passed them to the export.
    """
    authors = [
        {"name": name, **({"affiliation": metadata.affiliations[i]} if i < len(metadata.affiliations) else {})}
        for i, name in enumerate(metadata.authors)
    ]
    overrides: dict[str, object] = {
        "authors": authors,
        "license": metadata.license,
        "cite": [{"text": text, "doi": doi} for text, doi in metadata.references],
    }
    if metadata.description.strip():
        overrides["description"] = metadata.description
    return overrides


def _is_power_of_2(value: int) -> bool:
    """Return whether *value* is a positive power of two."""
    return value > 0 and float(np.log2(value)).is_integer()


def receptive_field(star_dist: StarDist2D) -> list[tuple[int, int]]:
    """Return the network's receptive field around a centre pixel (1.20.8's ``_my_compute_receptive_field``).

    1.20.8 sets ``model._tile_overlap`` to this before exporting, as a TensorFlow 2 workaround.
    """
    from scipy.ndimage import zoom

    size = tuple(g * 128 for g in star_dist.config.grid)
    if not all(_is_power_of_2(s) for s in size):
        raise ValueError(f"the receptive field needs power-of-two sizes, got {size}")
    middle = tuple(s // 2 for s in size)
    impulse = np.zeros((1, *size, star_dist.config.n_channel_in), dtype=np.float32)
    impulse[(0, *middle, slice(None))] = 1
    response = star_dist.predict(np.squeeze(impulse))[0]
    baseline = star_dist.predict(np.squeeze(np.zeros_like(impulse)))[0]
    grid = tuple((np.array(impulse.shape[1:-1]) / np.array(response.shape)).astype(int))
    changed = np.where(np.abs(zoom(response, grid, order=0) - zoom(baseline, grid, order=0)) > 0)
    return [(int(m - np.min(i)), int(np.max(i) - m)) for m, i in zip(middle, changed, strict=True)]


def export_model(star_dist: StarDist2D, output: Path, example_image: np.ndarray, metadata: ModelMetadata) -> Path:
    """Write the BioImage Model Zoo zip of *star_dist* to *output* (cell 38).

    1.20.8 names it ``<model folder>/<model name>bioimageio.zip`` (no separator); :func:`default_output` keeps that.
    """
    from stardist import export_bioimageio

    star_dist._tile_overlap = receptive_field(star_dist)
    export_bioimageio(
        star_dist, output, example_image, name=metadata.name, overwrite_spec_kwargs=spec_overrides(metadata)
    )
    return output


def default_output(model_folder: Path) -> Path:
    """Return 1.20.8's file name for the export: ``<parent>/<model name>bioimageio.zip``."""
    return model_folder.parent / f"{model_folder.name}bioimageio.zip"
