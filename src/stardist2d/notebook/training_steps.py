"""Notebook sections 3.2–4.2: training parameters, augmentation, pretrained weights, training (1.20.8 cells 20–30)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from IPython.display import display

import stardist2d
from stardist2d import config, data, environment, model, plotting, pretrained, report, train
from stardist2d.notebook.forms import Field, Form, message, show_figure
from stardist2d.notebook.session import Session

NOTEBOOK_VERSION = stardist2d.__version__
"""The version written in the training report: the package's, so it moves with every release."""
"""Version written in the reports (the notebook's ``current_version``)."""


def _settings_from(values: dict[str, Any]) -> config.TrainingSettings:
    """Return the training settings of the section 3.2 form."""
    return config.TrainingSettings(
        number_of_epochs=values["number_of_epochs"],
        use_default_advanced_parameters=values["Use_Default_Advanced_Parameters"],
        batch_size=values["batch_size"],
        number_of_steps=values["number_of_steps"],
        patch_size=values["patch_size"],
        percentage_validation=values["percentage_validation"],
        n_rays=values["n_rays"],
        grid_parameter=values["grid_parameter"],
        initial_learning_rate=values["initial_learning_rate"],
    )


def training_parameters(session: Session) -> Form:
    """Section 3.2: paths, model name and training parameters; shows a training pair (cell 20)."""

    def run(values: dict[str, Any]) -> None:
        if not values["model_name"].strip():
            raise ValueError("give the model a name (model_name)")
        session.training_source = session.path(values["Training_source"])
        session.training_target = session.path(values["Training_target"])
        session.model_folder = session.path(values["model_path"]) / values["model_name"].strip()
        session.settings = config.with_defaults(_settings_from(values))
        if values["Use_Default_Advanced_Parameters"]:
            message("Default advanced parameters enabled")
        pairs = data.training_pairs(session.training_source, session.training_target)
        image, labels = data.read_image(pairs[0][0]), data.read_image(pairs[0][1])
        message(f"Loaded images (width, length) = {image.shape}")
        for note in config.decide_patch_size(session.settings, image.shape).notes:
            message(note, warning=True)
        if session.model_folder.exists():
            message(f"!! WARNING: {session.model_folder.name} already exists: tick the box in 4.1 to replace it", True)
        session.example_figure = session.base_folder / "TrainingDataExample_StarDist2D.png"
        show_figure(plotting.training_pair(image, labels, session.example_figure))
        message("Parameters initiated.")

    fields = [
        Field("Training_source", "data/Stardist/Training - Images"),
        Field("Training_target", "data/Stardist/Training - Masks"),
        Field("model_name", ""),
        Field("model_path", "models"),
        Field("number_of_epochs", 100, "int"),
        Field("Use_Default_Advanced_Parameters", True, "bool"),
        Field("batch_size", 2, "int"),
        Field("number_of_steps", 50, "int"),
        Field("patch_size", 512, "int"),
        Field("percentage_validation", 10, "int"),
        Field("n_rays", 32, "int"),
        Field("grid_parameter", 2, "choice", config.GRID_CHOICES),
        Field("initial_learning_rate", 0.0003, "float"),
    ]
    return Form("Setting main training parameters", fields, run).show()


def data_augmentation(session: Session) -> Form:
    """Section 3.3: augmentation on or off and the dataset factor (cell 23)."""

    def run(values: dict[str, Any]) -> None:
        enabled = values["Use_Data_augmentation"]
        session.augmentation = config.AugmentationSettings(enabled, values["Multiply_dataset_by"])
        message("Data augmentation enabled" if enabled else "Data augmentation disabled", warning=not enabled)

    fields = [Field("Use_Data_augmentation", True, "bool"), Field("Multiply_dataset_by", 4, "slider", (1, 10, 1))]
    return Form("Data augmentation", fields, run).show()


def pretrained_weights(session: Session) -> Form:
    """Section 3.4: start from the weights of a pretrained network (cell 25)."""

    def run(values: dict[str, Any]) -> None:
        session.pretrained_weights = session.pretrained_learning_rate = None
        if not values["Use_pretrained_model"]:
            message("No pretrained network will be used.", warning=True)
            return
        choice = values["pretrained_model_choice"]
        if choice == "Model_from_file":
            folder = session.path(values["pretrained_model_path"])
        else:
            folder = pretrained.fetch_model_folder(choice, session.base_folder, values["bioimageio_model"])
        session.pretrained_weights = pretrained.weights_file(choice, folder, values["Weights_choice"])
        rate, note = pretrained.learning_rate(folder, values["Weights_choice"], session.settings.initial_learning_rate)
        session.pretrained_learning_rate = rate
        message(f"Weights found in: {session.pretrained_weights} will be loaded prior to training ({note}: {rate}).")

    fields = [
        Field("Use_pretrained_model", False, "bool"),
        Field(
            "pretrained_model_choice", "2D_versatile_fluo_from_Stardist_Fiji", "choice", pretrained.PRETRAINED_CHOICES
        ),
        Field("Weights_choice", "best", "choice", pretrained.WEIGHTS_CHOICES),
        Field("pretrained_model_path", ""),
        Field("bioimageio_model", ""),
    ]
    return Form("Loading weights from a pre-trained network", fields, run).show()


def prepare_training(session: Session) -> Form:
    """Section 4.1: read and split the data, create the model, load pretrained weights (cell 28)."""

    def run(values: dict[str, Any]) -> None:
        if session.model_folder is None or session.training_source is None or session.training_target is None:
            raise ValueError("run section 3.2 first")
        settings = session.settings
        if session.pretrained_learning_rate is not None:
            settings = replace(settings, initial_learning_rate=session.pretrained_learning_rate)
        session.training = train.load_training_data(
            session.training_source, session.training_target, settings, session.augmentation
        )
        if session.model_folder.exists() and values["overwrite_existing_model"]:
            message(f"!! WARNING: {session.model_folder} already exists and has been removed !!", True)
        session.model = train.new_model(session.training, session.model_folder, values["overwrite_existing_model"])
        if session.pretrained_weights is not None:
            model.load_weights(session.model, session.pretrained_weights)
        split = session.training.split
        total = len(split.x_train) + len(split.x_val)
        message(f"number of images: {total} — training: {len(split.x_train)}, validation: {len(split.x_val)}")
        if model.object_larger_than_field_of_view(session.model, split.y_train):
            message("WARNING: median object size larger than field of view of the neural network.", True)
        _training_report(session)
        message(f"Number of steps: {session.training.steps_per_epoch}")

    fields = [Field("overwrite_existing_model", False, "bool", help="Delete an existing model folder of the same name")]
    return Form("Create the model and dataset objects", fields, run).show()


def _training_report(session: Session) -> None:
    """Write the training report PDF (before and after training, as 1.20.8)."""
    if session.training is None or session.model_folder is None or session.model is None:
        raise ValueError("run section 4.1 first")
    training = session.training
    report.write_training_report(
        report.TrainingReport(
            session.model_folder,
            NOTEBOOK_VERSION,
            training.settings,
            session.augmentation,
            training.patch_size,
            training.steps_per_epoch,
            len(training.split.x_train) + len(training.split.x_val),
            training.image_shape,
            session.training_source or session.base_folder,
            session.training_target or session.base_folder,
            session.model.config.train_dist_loss,
            session.pretrained_weights is not None,
            environment.collect(),
            session.example_figure,
            session.training_seconds,
        )
    )


def start_training(session: Session) -> Form:
    """Section 4.2: train, optimise the thresholds, write the loss table and the report (cell 30)."""

    def run(values: dict[str, Any]) -> None:
        _ = values
        if session.training is None or session.model is None or session.model_folder is None:
            raise ValueError("run section 4.1 first")
        epochs = session.training.settings.number_of_epochs
        progress = message_line()

        def on_epoch(epoch: int, logs: dict[str, float]) -> None:
            loss, val_loss = logs.get("loss", 0.0), logs.get("val_loss", 0.0)
            progress(f"Epoch {epoch + 1}/{epochs}: loss {loss:.4f}, val_loss {val_loss:.4f}")

        augment = session.augmentation.use_data_augmentation
        result = train.train(session.model, session.training, epochs, augment, on_epoch)
        train.write_training_evaluation(session.model_folder, result.history)
        session.training_seconds = result.seconds
        hours, minutes, seconds = train.elapsed(result.seconds)
        message(f"Training done. Time elapsed: {hours} hour(s) {minutes} min(s) {seconds} sec(s)")
        _training_report(session)

    return Form("Start training", [], run, "Start training").show()


def message_line() -> Callable[[str], None]:
    """Return a function that rewrites one line of text in place (training progress)."""
    import ipywidgets as widgets

    line = widgets.HTML()
    display(line)

    def update(text: str) -> None:
        line.value = text

    return update
