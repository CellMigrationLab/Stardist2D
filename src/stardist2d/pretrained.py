"""Weights to start training from (1.20.8, section 3.4, cell 25).

The downloads go to a folder the caller chooses (the notebook uses its working folder), never into
the app's installation folder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from stardist2d import model_catalogue

PRETRAINED_CHOICES = (
    "Model_from_file",
    "BioImage Model Zoo",
    "2D_versatile_fluo_from_Stardist_Fiji",
    "2D_Demo_Model_from_Stardist_Github",
    "Versatile_H&E_nuclei",
    *(entry.label for entry in model_catalogue.CATALOGUE),
)
"""The choices of section 3.4: 1.20.8's, then the rehosted lab and Model Zoo models (1.2.0)."""
WEIGHTS_CHOICES = ("last", "best")

REGISTERED = {
    "2D_Demo_Model_from_Stardist_Github": "2D_demo",
    "2D_versatile_fluo_from_Stardist_Fiji": "2D_versatile_fluo",
    "Versatile_H&E_nuclei": "2D_versatile_he",
    **{entry.label: entry.name for entry in model_catalogue.CATALOGUE},
}
"""Section 3.4 choice -> StarDist's registered pretrained model.

Difference from 1.20.8: 1.20.8 downloaded these from cloud.mpi-cbg.de and from the old mpicbg-csbd GitHub path, which
no longer serve them (tester report, 1.1.1). They now come through ``StarDist2D.from_pretrained``: the StarDist team's
stardist-models GitHub releases, checked against StarDist's SHA-256. Those archives hold only ``weights_best.h5``."""
MODEL_FILES = ("config.json", "thresholds.json", "weights_best.h5")


@dataclass(frozen=True)
class PretrainedWeights:
    """The weights file to load and the learning rate to continue with."""

    weights_file: Path
    learning_rate: float
    learning_rate_note: str


def fetch_model_folder(choice: str, download_folder: Path, bioimageio_model: str = "") -> Path:
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
    if choice not in REGISTERED:
        raise ValueError(f"unknown pretrained model {choice!r}: choose one of {PRETRAINED_CHOICES}")
    return _copy_registered(REGISTERED[choice], download_folder / REGISTERED[choice])


def _copy_registered(name: str, folder: Path) -> Path:
    """Fetch StarDist's registered model *name* (downloaded once, checksum-verified) and copy its files to *folder*.

    Raises
    ------
    RuntimeError
        If the model cannot be downloaded, with where to get it by hand.
    """
    import shutil

    from stardist2d import model

    try:
        source = Path(model.from_registry(name).logdir)
    except Exception as problem:  # Keras' get_file wraps every download failure in a plain Exception
        raise RuntimeError(
            f"the pretrained model {name} could not be downloaded ({problem}). Download it from "
            "https://github.com/stardist/stardist-models/releases, unzip it and choose Model_from_file"
        ) from None
    folder.mkdir(parents=True, exist_ok=True)
    for file_name in MODEL_FILES:
        shutil.copy2(source / file_name, folder / file_name)
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
    elif choice in REGISTERED:
        name = "weights_best.h5"  # the stardist-models archives hold only the best weights
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
