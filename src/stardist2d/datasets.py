"""StarDist 2D datasets on Zenodo that section 3.1 can download (CC BY 4.0, Cell Migration Lab).

Each record is one zip with training images and masks, quality-control images and masks, and the StarDist
model trained on them; the catalogue says where each part is inside the zip, so the notebook can tell the
person which paths to use in sections 3.2, 5 and 6.
"""

from __future__ import annotations

import shutil
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Dataset:
    """One Zenodo record: its zip and the folders inside it (relative to the extraction folder)."""

    title: str
    record: str
    zip_name: str
    training_images: str
    training_masks: str
    qc_images: str
    qc_masks: str
    model: str

    @property
    def url(self) -> str:
        """Return the download URL of the zip."""
        return f"https://zenodo.org/records/{self.record}/files/{self.zip_name}?download=1"

    @property
    def doi(self) -> str:
        """Return the record's DOI."""
        return f"10.5281/zenodo.{self.record}"


def _dataset(title: str, record: str, zip_name: str, root: str, parts: tuple[str, str, str, str, str]) -> Dataset:
    """Return a catalogue entry whose folders all sit under *root*."""
    return Dataset(title, record, zip_name, *(f"{root}/{part}" for part in parts))


CATALOGUE = (
    _dataset(
        "HUVEC nuclei (fluorescence)",
        "10617532",
        "StarDist_HUVEC_nuclei_dataset.zip",
        "StarDist_HUVEC_nuclei_dataset_OK",
        (
            "Training_data/Training_source",
            "Training_data/Training_target",
            "Quality_control_data/QC_source",
            "Quality_control_data/QC_target",
            "210916_stardist_HUVEC_nuclei_model",
        ),
    ),
    _dataset(
        "Neutrophils (brightfield)",
        "10572231",
        "StarDist_BF_Neutrophils_dataset.zip",
        "StarDist_BF_Neutrophils_dataset_OK",
        (
            "Training_data/Training_sources",
            "Training_data/Training_targets",
            "Quality_control_data/Neutrophils_QC_Sources",
            "Quality_control_data/Neutrophils_QC_targets",
            "Stardist_Neutrophils_model",
        ),
    ),
    _dataset(
        "Monocytes (brightfield)",
        "10572200",
        "StarDist_BF_Monocytes_dataset.zip",
        "StarDist_BF_Monocytes_dataset_OK",
        (
            "Training_data/Training_sources",
            "Training_data/Training_targets",
            "Quality_control_data/Monocytes_QC_sources",
            "Quality_control_data/Monocytes_QC_Targets",
            "Stardist_Monocytes_model",
        ),
    ),
    _dataset(
        "Cancer cells 20x (brightfield)",
        "10572122",
        "StarDist_BF_Cancer_cell_dataset.zip",
        "StarDist_BF_Cancer_cell_dataset_OK",
        (
            "Training_data/230313_Stardist_CC_round_source_JP",
            "Training_data/230313_Stardist_CC_round_target_JP",
            "Quality_control_data/230313_Stardist_CC_QA_round_source_JP",
            "Quality_control_data/230313_Stardist_CC_QA_round_target_JP",
            "Stardist_CancerCells_model",
        ),
    ),
    _dataset(
        "Cancer cells 10x (brightfield)",
        "13304399",
        "StarDist_BF10x_Cancer_cell_dataset.zip",
        "StarDist_BF10x_Cancer_cell_dataset",
        (
            "Training_data/Training_source",
            "Training_data/Training_target",
            "QC_data/Input",
            "QC_data/Output",
            "StarDist_model",
        ),
    ),
    _dataset(
        "Fluorescent cells",
        "10572310",
        "StarDist_Fluorescent_cells.zip",
        "StarDist_Fluorescent_cells_OK",
        (
            "Training dataset/Training_Source",
            "Training dataset/Training_target",
            "QC_data/QC_source",
            "QC_data/QC_target",
            "StarDist Model",
        ),
    ),
    _dataset(
        "AsPC1 cells (Lifeact)",
        "13442128",
        "Stardist_AsPC1_from _TC.zip",
        "Stardist_AsPC1_from _TC",
        (
            "Training_data/Training_source",
            "Training_data/Training_targets",
            "Quality_control_data/QC_source",
            "Quality_control_data/QC_targets",
            "231222_stardist_AsPC1_from _TC-ch_MODEL",
        ),
    ),
    _dataset(
        "MiaPaCa2 cells (CD44)",
        "13442877",
        "Stardist_MiaPaCa2_from_CD44.zip",
        "Stardist_MiaPaCa2_from_CD44",
        (
            "Training_data/Training_source",
            "Training_data/Training_targets",
            "Quality_control_data/QC_source",
            "Quality_control_data/QC_targets",
            "240116_Stardist_MiaPaCa2_from_CD44_ch_MODEL",
        ),
    ),
    _dataset(
        "Tumour cell nuclei (DAPI)",
        "13443221",
        "Stardist_TCs_nuclei_DAPI.zip",
        "Stardist_TCs_nuclei_DAPI",
        (
            "Training_data/Training_sources (1)",
            "Training_data/Training_targets (1)",
            "Quality_control_data/QC_source",
            "Quality_control_data/QC_targets",
            "231027_Stardist_TCs_nuclei_DAPI-ch_MODEL",
        ),
    ),
)
"""The datasets, as listed on Zenodo (sizes 100–580 MB)."""


def by_title(title: str) -> Dataset:
    """Return the catalogue entry called *title*.

    Raises
    ------
    ValueError
        If no dataset has that title.
    """
    for dataset in CATALOGUE:
        if dataset.title == title:
            return dataset
    raise ValueError(f"unknown dataset {title!r}: choose one of {[d.title for d in CATALOGUE]}")


def download(dataset: Dataset, destination: Path, progress: Callable[[float, str], None] | None = None) -> Path:
    """Download and unpack *dataset* into ``destination/<record>`` and return that folder.

    A dataset already unpacked there is not downloaded again.

    Raises
    ------
    RuntimeError
        If the download fails.
    """
    import requests

    folder = destination / dataset.record
    if (folder / dataset.training_images).is_dir():
        return folder
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / dataset.zip_name
    with requests.get(dataset.url, stream=True, timeout=600) as response:
        if response.status_code != 200:
            raise RuntimeError(
                f"{dataset.title} could not be downloaded from {dataset.url} (HTTP {response.status_code})"
            )
        total, done = int(response.headers.get("Content-Length", 0)), 0
        with archive.open("wb") as handle:
            for block in response.iter_content(1 << 20):
                handle.write(block)
                done += len(block)
                if progress and total:
                    progress(done / total, f"{dataset.title}: {done / 1e6:.0f} of {total / 1e6:.0f} MB")
    with zipfile.ZipFile(archive) as unpacked:
        unpacked.extractall(folder, [name for name in unpacked.namelist() if "__MACOSX" not in name])
    archive.unlink()
    shutil.rmtree(folder / "__MACOSX", ignore_errors=True)
    return folder
