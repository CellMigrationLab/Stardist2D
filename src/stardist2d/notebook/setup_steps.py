"""Notebook sections 1–3.1: dependencies, GPU, Google Drive, example data (1.20.8 cells 8, 11, 13, 17)."""

from __future__ import annotations

from typing import Any

from IPython.display import display

import stardist2d
from stardist2d import data, environment
from stardist2d.notebook.forms import Field, Form, message
from stardist2d.notebook.session import Session, on_colab


def load_dependencies() -> Session:
    """Section 1.3: import the package, show the versions and return a new session."""
    info = environment.collect()
    versions = ", ".join(f"{name} {version}" for name, version in info.versions.items())
    message(f"StarDist2D {stardist2d.__version__} ready. Libraries: {versions}.")
    return Session()


def check_gpu() -> None:
    """Section 2.1: show the GPU (``nvidia-smi``) or say training will use the CPU."""
    report = environment.gpu_report()
    if report.startswith("You do not have a GPU"):
        message(report, warning=True)
    else:
        print(report)


def mount_drive() -> None:
    """Section 2.2: on Colab, connect Google Drive at ``/content/gdrive``; locally, nothing to do."""
    if not on_colab():
        message("Running locally: your files are already available, there is no Google Drive to connect.")
        return
    from google.colab import drive

    drive.mount("/content/gdrive", force_remount=True)


def download_example_data(session: Session) -> Form:
    """Section 3.1: put the example dataset in the base folder (the shipped copy offline, Zenodo otherwise)."""

    def run(values: dict[str, Any]) -> None:
        session.base_folder = session.path(values["base_folder"])
        target = data.get_example_data(session.base_folder / "data", full=values["download_full_dataset"])
        message(f"Example dataset ready in {target}")
        display({"text/plain": "\n".join(sorted(p.relative_to(target).as_posix() for p in target.iterdir()))}, raw=True)

    fields = [
        Field("base_folder", str(session.base_folder), help="Relative paths in the next sections start here"),
        Field("download_full_dataset", False, "bool", help="Download all 45 training pairs from Zenodo (internet)"),
    ]
    return Form("Download a test dataset", fields, run).show()
