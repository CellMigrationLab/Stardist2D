"""What the notebook's steps share (1.20.8 kept it in notebook globals: ``model_name``, ``conf``, ``Z``...)."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from stardist2d import config

if TYPE_CHECKING:
    from stardist.models import StarDist2D

    from stardist2d.train import TrainingData


def on_colab() -> bool:
    """Return whether the notebook runs on Google Colab."""
    return "google.colab" in sys.modules


def default_base_folder() -> Path:
    """Return where relative paths start: ``/content`` on Colab, ``~/StarDist2D`` locally (1.20.8: ``base_path``)."""
    return Path("/content") if on_colab() else Path.home() / "StarDist2D"


@dataclass
class Session:
    """State passed from one notebook step to the next."""

    base_folder: Path = field(default_factory=default_base_folder)
    training_source: Path | None = None
    training_target: Path | None = None
    model_folder: Path | None = None
    settings: config.TrainingSettings = field(default_factory=config.TrainingSettings)
    augmentation: config.AugmentationSettings = field(default_factory=config.AugmentationSettings)
    pretrained_weights: Path | None = None
    pretrained_learning_rate: float | None = None
    training: TrainingData | None = None
    model: StarDist2D | None = None
    training_seconds: float | None = None
    qc_model_folder: Path | None = None
    qc_rows: list[dict[str, Any]] = field(default_factory=list)
    qc_images: Path | None = None
    qc_masks: Path | None = None
    example_figure: Path | None = None

    def path(self, text: str) -> Path:
        """Return *text* as a path; a relative path starts from :attr:`base_folder` (1.20.8's ``check_base_path``).

        Raises
        ------
        ValueError
            If *text* is empty.
        """
        if not text.strip():
            raise ValueError("a folder or file path is empty: fill it in above")
        path = Path(text.strip()).expanduser()
        return path if path.is_absolute() else self.base_folder / path

    def model_from_choice(self, use_current: bool, folder_text: str) -> Path:
        """Return the model folder: the one trained in this session, or the folder typed in (sections 5 and 6).

        Raises
        ------
        ValueError
            If the current model is asked for but none was trained in this session.
        FileNotFoundError
            If the folder is not a StarDist model folder.
        """
        if use_current:
            if self.model_folder is None:
                raise ValueError("no model was trained in this session: untick the box and give a model folder")
            folder = self.model_folder
        else:
            folder = self.path(folder_text)
        if not (folder / "config.json").is_file():
            raise FileNotFoundError(f"{folder} is not a StarDist model folder (config.json is missing)")
        return folder
