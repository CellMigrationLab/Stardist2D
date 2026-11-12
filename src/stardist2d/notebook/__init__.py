"""The notebook's steps: one function per code cell of ZeroCostDL4Mic StarDist 2D 1.20.8.

Each cell of ``notebooks/StarDist_2D/StarDist_2D_ZeroCostDL4Mic.ipynb`` calls one of these functions, which
shows the cell's form (same parameter names and defaults as 1.20.8) and runs the step with the package.
"""

from __future__ import annotations

from stardist2d.notebook.evaluation_steps import (
    choose_qc_model,
    export_bioimage_model_zoo,
    loss_curves,
    quality_control,
)
from stardist2d.notebook.prediction_steps import big_data_predictions, predictions
from stardist2d.notebook.session import Session, on_colab
from stardist2d.notebook.setup_steps import check_gpu, download_example_data, load_dependencies, mount_drive
from stardist2d.notebook.training_steps import (
    data_augmentation,
    prepare_training,
    pretrained_weights,
    start_training,
    training_parameters,
)

__all__ = [
    "Session",
    "big_data_predictions",
    "check_gpu",
    "choose_qc_model",
    "data_augmentation",
    "download_example_data",
    "export_bioimage_model_zoo",
    "load_dependencies",
    "loss_curves",
    "mount_drive",
    "on_colab",
    "predictions",
    "prepare_training",
    "pretrained_weights",
    "quality_control",
    "start_training",
    "training_parameters",
]
