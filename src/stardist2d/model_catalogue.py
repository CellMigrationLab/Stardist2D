"""StarDist 2D models of the Cell Migration Lab (and one from the BioImage Model Zoo), rehosted in this repository.

Each model is a small zip in ``models/`` (config.json, thresholds.json, weights_best.h5 and a README naming its source
and licence), registered in StarDist's own pretrained registry: ``StarDist2D.from_pretrained(name)`` downloads it once,
checks its SHA-256 and caches it, exactly like StarDist's ``2D_versatile_fluo``. The lab's models were taken unchanged
from the model folders of its Zenodo datasets (section 3.1 downloads the whole datasets); the CHO model from its
BioImage Model Zoo entry. All are single-channel, 32 rays, grid 2.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

MODELS_TAG = "1.2.0"
"""The release whose models/ folder the zips come from (the files never change, and their SHA-256 is checked)."""
URL_VARIABLE = "STARDIST2D_MODELS_URL"
"""Environment variable that replaces the download address (tests: the repository's own ``models/`` as a file URL)."""


@dataclass(frozen=True)
class CatalogueModel:
    """One rehosted model: registry name, label shown in forms, image type, authors, what to cite, licence, SHA-256."""

    name: str
    label: str
    kind: str
    authors: str
    cite: str
    licence: str
    sha256: str


CATALOGUE = (
    CatalogueModel(
        "cml_huvec_nuclei",
        "CML: HUVEC nuclei (fluorescence)",
        "fluorescence",
        "Follain, Gautier, Ghimire, Sujan, Pylvänäinen, Joanna, Ivaska, Johanna, Jacquemet, Guillaume",
        "doi:10.5281/zenodo.10617532",
        "cc-by-4.0",
        "47a5f56f83e718bf46b40eb1681dbc09b9af7b895fa7f6dd4913284176f71fd8",
    ),
    CatalogueModel(
        "cml_fluorescent_cells",
        "CML: fluorescent cells",
        "fluorescence",
        "Follain, Gautier, Ghimire, Sujan, Pylvänäinen, Joanna, Ivaska, Johanna, Jacquemet, Guillaume",
        "doi:10.5281/zenodo.10572310",
        "cc-by-4.0",
        "3137961901aaef20d62cdbc6b9022ffb583515fb738d23de4569bb305f3625bc",
    ),
    CatalogueModel(
        "cml_aspc1_lifeact",
        "CML: AsPC1 cells, Lifeact (fluorescence)",
        "fluorescence",
        "Follain, Gautier, Ghimire, Sujan, Pylvänäinen, Joanna, Ivaska, Johanna, Jacquemet, Guillaume",
        "doi:10.5281/zenodo.13442128",
        "cc-by-4.0",
        "901b36a42736391e019b6fa46b2087cc565e5f949abe1334f9675e3bd9b68f7b",
    ),
    CatalogueModel(
        "cml_miapaca2_cd44",
        "CML: MiaPaCa2 cells, CD44 (fluorescence)",
        "fluorescence",
        "Follain, Gautier, Ghimire, Sujan, Pylvänäinen, Joanna, Ivaska, Johanna, Jacquemet, Guillaume",
        "doi:10.5281/zenodo.13442877",
        "cc-by-4.0",
        "b3139f57c1b3eb0396dd2dbf8c2c842a6203454cdea2fefce42b548851e68746",
    ),
    CatalogueModel(
        "cml_tumour_cell_nuclei_dapi",
        "CML: tumour cell nuclei, DAPI (fluorescence)",
        "fluorescence",
        "Follain, Gautier, Ghimire, Sujan, Pylvänäinen, Joanna, Ivaska, Johanna, Jacquemet, Guillaume",
        "doi:10.5281/zenodo.13443221",
        "cc-by-4.0",
        "e655d37d941c4a44db54458cc187d31d96dc3607c8958ecc6d41f92a8b394cff",
    ),
    CatalogueModel(
        "cml_neutrophils_bf",
        "CML: neutrophils (brightfield)",
        "brightfield",
        "Follain, Gautier, Ghimire, Sujan, Pylvänäinen, Joanna, Ivaska, Johanna, Jacquemet, Guillaume",
        "doi:10.5281/zenodo.10572231",
        "cc-by-4.0",
        "2b60ebd7e91feb47c0cfadaaea032da5c14b7a6289837cf20d638291559db369",
    ),
    CatalogueModel(
        "cml_monocytes_bf",
        "CML: monocytes (brightfield)",
        "brightfield",
        "Follain, Gautier, Ghimire, Sujan, Pylvänäinen, Joanna, Ivaska, Johanna, Jacquemet, Guillaume",
        "doi:10.5281/zenodo.10572200",
        "cc-by-4.0",
        "eac8ab3f31634c69b0003df15f6a532349344d7fe6650b1264380d1cbd1b8385",
    ),
    CatalogueModel(
        "cml_cancer_cells_bf_20x",
        "CML: cancer cells 20x (brightfield)",
        "brightfield",
        "Follain, Gautier, Ghimire, Sujan, Pylvänäinen, Joanna, Ivaska, Johanna, Jacquemet, Guillaume",
        "doi:10.5281/zenodo.10572122",
        "cc-by-4.0",
        "d648ae903db074f3d1240aa03b78ba0481bc627d9f380a93174d7cd9c3cad206",
    ),
    CatalogueModel(
        "cml_cancer_cells_bf_10x",
        "CML: cancer cells 10x (brightfield)",
        "brightfield",
        "Follain, Gautier, Ghimire, Sujan, Pylvänäinen, Joanna, Ivaska, Johanna, Jacquemet, Guillaume",
        "doi:10.5281/zenodo.13304399",
        "cc-by-4.0",
        "97d5ff4eb4e2b0f267d7c4a2c7b57e09f9e3acca9002cf2120623db4a7e06d50",
    ),
    CatalogueModel(
        "bmz_cho_mitotic_rounding_bf",
        "BMZ: CHO mitotic rounding (brightfield)",
        "brightfield",
        "Estibaliz Gómez de Mariscal, Mario Del Rosario",
        "BioImage Model Zoo entry sincere-microbe (https://bioimage.io/#/?id=sincere-microbe)",
        "MIT",
        "ac86a1954e5153e03b44baec330fae933b12a77040b4e97aa6e829be06fbfd62",
    ),
)


def base_url() -> str:
    """Return the address of the ``models/`` folder the zips are downloaded from."""
    default = f"https://raw.githubusercontent.com/CellMigrationLab/Stardist2D/{MODELS_TAG}/models/"
    return os.environ.get(URL_VARIABLE, default).rstrip("/") + "/"


def by_label(label: str) -> CatalogueModel | None:
    """Return the catalogue model shown as *label* (or named *label*), or None."""
    return next((m for m in CATALOGUE if label in (m.label, m.name)), None)


def citation(label: str) -> str:
    """Return what to cite for a catalogue model, or "" for any other model."""
    entry = by_label(label)
    return f"{entry.label}: please cite {entry.cite} ({entry.authors}; licence {entry.licence})" if entry else ""
