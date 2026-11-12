"""Notebook sections 1–3.1: dependencies, GPU, Google Drive, example data (1.20.8 cells 8, 11, 13, 17)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from IPython.display import display

import stardist2d
from stardist2d import data, datasets, environment
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


EXAMPLE = "Example: nuclei (shipped, works offline)"


def _show_paths(session: Session, rows: list[tuple[str, Path]]) -> None:
    """Show which paths to type in the next sections (relative to the base folder when possible)."""

    def shown(path: Path) -> str:
        try:
            return path.relative_to(session.base_folder).as_posix()
        except ValueError:
            return str(path)

    lines = [f"{label:<34}{shown(path)}" for label, path in rows]
    display({"text/plain": "\n".join(lines)}, raw=True)


def download_example_data(session: Session) -> Form:
    """Section 3.1: the example dataset (shipped copy offline, or the full record), or one of the Zenodo datasets."""

    def run(values: dict[str, Any]) -> None:
        session.base_folder = session.path(values["base_folder"])
        if values["dataset"] == EXAMPLE:
            target = data.get_example_data(session.base_folder / "data", full=values["download_full_dataset"])
            message(f"Example dataset ready in {target}")
            rows = [
                ("Training_source / Training_target:", target / "Training - Images"),
                ("", target / "Training - Masks"),
                ("Source_QC_folder / Target_QC_folder:", target / "Test - Images"),
                ("", target / "Test - Masks"),
            ]
            _show_paths(session, rows)
            return
        dataset = datasets.by_title(values["dataset"])
        message(f"Downloading {dataset.title} (Zenodo {dataset.doi}, CC BY 4.0)…")
        folder = datasets.download(dataset, session.base_folder / "data", progress=message_line())
        message(f"{dataset.title} ready in {folder}")
        _show_paths(
            session,
            [
                ("Training_source:", folder / dataset.training_images),
                ("Training_target:", folder / dataset.training_masks),
                ("Source_QC_folder:", folder / dataset.qc_images),
                ("Target_QC_folder:", folder / dataset.qc_masks),
                ("Trained model (QC / predictions):", folder / dataset.model),
            ],
        )

    fields = [
        Field("base_folder", str(session.base_folder), help="Relative paths in the next sections start here"),
        Field(
            "dataset",
            EXAMPLE,
            "choice",
            (EXAMPLE, *(d.title for d in datasets.CATALOGUE)),
            help="The other datasets are downloaded from Zenodo (100-580 MB, internet needed)",
        ),
        Field("download_full_dataset", False, "bool", help="Example only: download all 45 training pairs from Zenodo"),
    ]
    return Form("Download a test dataset", fields, run).show()


def message_line() -> Callable[[float, str], None]:
    """Return a progress callback that rewrites one line of text in place."""
    import ipywidgets as widgets

    line = widgets.HTML()
    display(line)

    def update(fraction: float, text: str) -> None:
        line.value = f"{fraction:4.0%} {text}"

    return update
