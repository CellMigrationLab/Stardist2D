"""Weights to start training from (1.20.8, section 3.4, cell 25).

The downloads go to a folder the caller chooses (the notebook uses its working folder), never into
the app's installation folder.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

PRETRAINED_CHOICES = (
    "Model_from_file",
    "BioImage Model Zoo",
    "2D_versatile_fluo_from_Stardist_Fiji",
    "2D_Demo_Model_from_Stardist_Github",
    "Versatile_H&E_nuclei",
)
"""The choices of section 3.4, as 1.20.8 names them."""
WEIGHTS_CHOICES = ("last", "best")

_DEMO = "https://github.com/mpicbg-csbd/stardist/raw/master/models/examples/2D_demo/"
_FIJI = "https://cloud.mpi-cbg.de/index.php/s/1k5Zcy7PpFWRb0Q/download?path=/versatile&files="
DOWNLOADS = {
    "2D_Demo_Model_from_Stardist_Github": (
        "2D_Demo",
        {name: _DEMO + name for name in ("config.json", "thresholds.json", "weights_best.h5", "weights_last.h5")},
    ),
    "2D_versatile_fluo_from_Stardist_Fiji": (
        "2D_versatile_fluo",
        {"2D_versatile_fluo.zip": _FIJI + "2D_versatile_fluo.zip"},
    ),
    "Versatile_H&E_nuclei": ("2D_versatile_he", {"2D_versatile_he.zip": _FIJI + "2D_versatile_he.zip"}),
}
"""Folder name and files of each downloadable model (the URLs of 1.20.8)."""


@dataclass(frozen=True)
class PretrainedWeights:
    """The weights file to load and the learning rate to continue with."""

    weights_file: Path
    learning_rate: float
    learning_rate_note: str


def _download(url: str, target: Path, timeout: float) -> Path:
    """Download *url* to *target*: streamed to a temporary file that replaces *target* only when complete.

    Raises
    ------
    RuntimeError
        If the server does not answer with the file, or sends fewer bytes than it announced.
    """
    import requests

    from stardist2d.export import atomic_write

    def fetch(temporary: Path) -> None:
        with requests.get(url, stream=True, timeout=timeout) as response:
            if response.status_code != 200:
                raise RuntimeError(f"could not download {url} (HTTP {response.status_code})")
            total, done = int(response.headers.get("Content-Length", 0)), 0
            with temporary.open("wb") as handle:
                for block in response.iter_content(1 << 20):
                    handle.write(block)
                    done += len(block)
        if total and done != total:
            raise RuntimeError(f"the download of {url} stopped after {done} of {total} bytes; try again")

    return atomic_write(target, fetch)


def _unpack(archive: Path, folder: Path) -> None:
    """Unpack *archive* into *folder* through a staging folder; refuse an archive without ``config.json`` and weights.

    Raises
    ------
    RuntimeError
        If the unpacked model lacks ``config.json`` or a ``weights_*.h5`` file.
    """
    import shutil

    staging = folder.with_name(f".{folder.name}.partial")
    shutil.rmtree(staging, ignore_errors=True)
    with zipfile.ZipFile(archive) as unpacked:
        unpacked.extractall(staging)
    model_folder = next((path.parent for path in staging.rglob("config.json")), None)
    if model_folder is None or not any(model_folder.glob("weights_*.h5")):
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError(f"{archive.name} does not contain a StarDist model (config.json and weights)")
    for path in staging.iterdir():
        target = folder / path.name
        if target.is_dir():
            shutil.rmtree(target)
        path.replace(target)
    staging.rmdir()


def fetch_model_folder(choice: str, download_folder: Path, bioimageio_model: str = "", timeout: float = 600) -> Path:
    """Download (or import) the model of a section 3.4 *choice* and return its folder.

    ``Model_from_file`` is not handled here: the caller already has the folder.

    Raises
    ------
    ValueError
        If *choice* is unknown, or a BioImage Model Zoo model is chosen without its id, URL or path.
    """
    if choice == "BioImage Model Zoo":
        if not bioimageio_model.strip():
            raise ValueError("give the BioImage Model Zoo id, URL or file of the model")
        from stardist import import_bioimageio

        folder = download_folder / "bioimage_model_zoo"
        import_bioimageio(bioimageio_model, folder)
        return folder
    if choice not in DOWNLOADS:
        raise ValueError(f"unknown pretrained model {choice!r}: choose one of {PRETRAINED_CHOICES}")
    name, files = DOWNLOADS[choice]
    folder = download_folder / name
    folder.mkdir(parents=True, exist_ok=True)
    for file_name, url in files.items():
        target = _download(url, folder / file_name, timeout)
        if target.suffix == ".zip":
            _unpack(target, folder)
    return folder


def weights_file(choice: str, model_folder: Path, weights: str) -> Path:
    """Return the weights file 1.20.8 loads for *choice* (cell 25).

    Raises
    ------
    FileNotFoundError
        If the file does not exist (1.20.8 printed a warning and trained from scratch instead).
    """
    if weights not in WEIGHTS_CHOICES:
        raise ValueError(f"weights must be one of {WEIGHTS_CHOICES}")
    if choice == "BioImage Model Zoo":
        name = "weights_bioimageio.h5"
    elif choice in ("2D_versatile_fluo_from_Stardist_Fiji", "Versatile_H&E_nuclei"):
        name = "weights_best.h5"
    else:
        name = f"weights_{weights}.h5"
    path = model_folder / name
    if not path.is_file():
        raise FileNotFoundError(f"the pretrained weights {path} do not exist")
    return path


def learning_rate(model_folder: Path, weights: str, default: float) -> tuple[float, str]:
    """Return the learning rate to continue training with, and a note saying where it came from (cell 25).

    From ``Quality Control/training_evaluation.csv``: the last value for ``last`` weights, the value at the
    lowest validation loss for ``best`` weights. Without that file or column, *default*.
    """
    import pandas as pd

    table_path = model_folder / "Quality Control" / "training_evaluation.csv"
    if not table_path.is_file():
        return default, f"no training_evaluation.csv: the learning rate {default} is used"
    table = pd.read_csv(table_path, sep=",")
    if "learning rate" not in table.columns:
        return default, f"the learning rate cannot be identified: {default} is used"
    if weights == "last":
        return float(table["learning rate"].iloc[-1]), "last learning rate of the pretrained network"
    best = table[table["val_loss"] == min(table["val_loss"])]
    return float(best["learning rate"].iloc[-1]), "learning rate of the best validation loss"
