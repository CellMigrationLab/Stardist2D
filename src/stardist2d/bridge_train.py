"""Training and folder quality control for the Fiji, napari and QuPath tools (LabConstrictor-Tools bridge).

The science is the notebook's (sections 3.2-5): ``train.load_training_data``, ``train.train``, ``pretrained`` and
``qc.run_quality_control``, with the same defaults and output files. What is added here is what a host needs: folder
checks with messages a person can act on, progress after every epoch, Cancel within a batch, and tables instead of
figures.

Cancel: the hosts stop the worker 3 s after Cancel, so nothing slow can run afterwards. What StarDist saved until then
stays in the model folder (``config.json``, ``weights_best.h5`` of the best epoch so far), with
``Quality Control/training_evaluation.csv`` and ``stardist2d_run.json`` rewritten after every epoch. Such a model works
with StarDist's default thresholds; its run file says where training stopped.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from stardist2d import config, data, environment, metrics, model, model_catalogue, pretrained, provenance, qc, train
from stardist2d.bridge import tool_error, tool_hooks

if TYPE_CHECKING:
    import pandas as pd
    from stardist.models import StarDist2D

    from stardist2d.prepare import TrainingSplit

START_CHOICES = (
    "None",
    "2D_versatile_fluo",
    "2D_versatile_he",
    "2D_paper_dsb2018",
    *(entry.label for entry in model_catalogue.CATALOGUE),
    "Own model folder",
)
"""What training can start from: nothing, a StarDist pretrained model, or a model folder (notebook section 3.4)."""
OWN_MODEL = "Own model folder"
CPU_NOTE = (
    "Training runs on the CPU on this computer: expect minutes per epoch. For long trainings use the notebook on "
    "Google Colab, or a Linux computer with an NVIDIA GPU."
)
LOW_F1 = 0.5
"""Below this validation F1 the training summary suggests more epochs or a pretrained start."""
_MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._ -]{0,99}")


@dataclass(frozen=True)
class TrainingRequest:
    """What the "Train a model" form asks for (the notebook's section 3.2 and 3.4 fields)."""

    images: Path
    masks: Path
    parent: Path
    name: str
    settings: config.TrainingSettings
    use_augmentation: bool = True
    start_from: str = "None"
    start_folder: Path | None = None
    overwrite: bool = False


def model_target(request: TrainingRequest) -> Path:
    """Return the new model folder ``parent/name``; refuse a bad name, an existing folder or an input folder."""
    name = request.name.strip()
    if not _MODEL_NAME.fullmatch(name) or ".." in name:
        raise tool_error(
            "invalid_parameter",
            f"model name {request.name!r}: use letters, digits, spaces, '.', '_' or '-' (no '/' or '\\')",
        )
    target = Path(request.parent) / name
    inputs = {Path(request.images).resolve(), Path(request.masks).resolve()}
    if target.resolve() in inputs or any(target.resolve() in folder.parents for folder in inputs):
        raise tool_error("invalid_parameter", "the model folder must not be, or contain, the training folders")
    if target.exists() and not request.overwrite:
        raise tool_error(
            "invalid_parameter",
            f"{target} already exists: choose another model name, or tick 'overwrite' to replace it",
        )
    return target


def start_weights(request: TrainingRequest) -> tuple[Path | None, float | None, str]:
    """Return the weights to start from, the learning rate to continue with (or None) and a note (section 3.4)."""
    if request.start_from not in START_CHOICES:
        raise tool_error("invalid_parameter", f"start from must be one of {START_CHOICES}")
    if request.start_from == "None":
        return None, None, "trained from scratch"
    try:
        if request.start_from == OWN_MODEL:
            if request.start_folder is None:
                raise FileNotFoundError("choose the model folder to start from")
            folder = Path(request.start_folder)
            rate, note = pretrained.learning_rate(folder, "best", request.settings.initial_learning_rate)
            return (
                pretrained.weights_file("Model_from_file", folder, "best"),
                rate,
                f"started from {folder.name}; {note}",
            )
        source = model.load_model(model.ModelChoice(request.start_from))
        cite = model_catalogue.citation(request.start_from)
        note = f"started from {request.start_from}; {cite}" if cite else f"started from StarDist's {request.start_from}"
        return Path(source.logdir) / "weights_best.h5", None, note
    except FileNotFoundError as problem:
        raise tool_error("invalid_parameter", str(problem)) from None


def _has_gpu() -> bool:
    """Return True when TensorFlow sees a GPU (Linux with an NVIDIA card)."""
    import tensorflow as tf

    return bool(tf.config.list_physical_devices("GPU"))


def _prepare(request: TrainingRequest, target: Path) -> tuple[train.TrainingData, StarDist2D, str]:
    """Read the training pairs, create the model and load the starting weights; refuse with what to do."""
    from labconstrictor_tools import progress

    weights, rate, note = start_weights(request)
    settings = request.settings if rate is None else replace(request.settings, initial_learning_rate=rate)
    progress(0.0, "reading the training images" + ("" if _has_gpu() else ". " + CPU_NOTE))
    try:
        training = train.load_training_data(
            Path(request.images), Path(request.masks), settings, config.AugmentationSettings(request.use_augmentation)
        )
    except (ValueError, FileNotFoundError) as problem:
        raise tool_error("invalid_parameter", str(problem)) from None
    star_dist = train.new_model(training, target, request.overwrite)
    if weights is not None:
        try:
            model.load_weights(star_dist, weights)
        except ValueError as problem:
            raise tool_error(
                "invalid_parameter",
                f"the starting model does not fit these settings (channels, n_rays or grid differ): {problem}",
            ) from None
    return training, star_dist, note


def _run_record(target: Path, request: TrainingRequest, inputs: list[Path], state: str) -> None:
    """Write the model folder's ``stardist2d_run.json`` with the settings and the training *state*."""
    settings = {"request": request, "state": state}
    provenance.write_run_file(target, "train (Fiji/napari/QuPath tool)", settings, inputs, environment.collect())


def train_model(request: TrainingRequest) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Train a model as notebook sections 4.1-4.2 do; return the loss table, the validation QC and a summary."""
    from labconstrictor_tools import check_cancel, progress

    target = model_target(request)
    training, star_dist, note = _prepare(request, target)
    epochs = training.settings.number_of_epochs
    inputs = [path for pair in data.training_pairs(Path(request.images), Path(request.masks)) for path in pair]
    _run_record(target, request, inputs, "started; no epoch finished")
    reporter = _TrainingProgress(epochs, training.steps_per_epoch, progress, check_cancel)

    def on_epoch(epoch: int, logs: dict[str, float]) -> None:
        reporter.epoch_done(epoch, logs)
        train.write_training_evaluation(target, reporter.history)
        _run_record(target, request, inputs, f"stopped after epoch {epoch + 1} of {epochs}; thresholds not optimised")

    result = train.train(star_dist, training, epochs, request.use_augmentation, on_epoch, reporter.batch_done)
    train.write_training_evaluation(target, result.history)
    progress(0.95, "quality control on the validation images")
    validation = validation_quality(result.model, training.split)
    _run_record(target, request, inputs, f"finished: {epochs} epochs, thresholds optimised")
    return _training_tables(result, validation, target, note)


class _TrainingProgress:
    """Progress of a training for the hosts: after every epoch (with its losses) and within an epoch (its steps).

    Steps are reported at most once a second: on a CPU an epoch can take minutes, and a host that shows nothing
    for that long looks frozen (tester report, 1.1.1); on a GPU, batches are too fast to report each. Every batch
    also checks Cancel (``check_cancel`` raises to stop training at once).
    """

    def __init__(
        self, epochs: int, steps: int, progress: Callable[[float, str], None], check_cancel: Callable[[], None]
    ) -> None:
        self.epochs, self.steps, self.progress, self.check_cancel = epochs, max(steps, 1), progress, check_cancel
        self.history: dict[str, list[float]] = {"loss": [], "val_loss": [], "learning_rate": []}
        self.epoch, self.step, self.last_report, self.losses = 0, 0, float("-inf"), ""

    def batch_done(self) -> None:
        """Count a batch, check Cancel and report the step when a second has passed since the last report."""
        self.check_cancel()
        self.step += 1
        now = time.monotonic()
        if now - self.last_report >= 1.0:
            self.last_report = now
            done = (self.epoch + min(self.step, self.steps) / self.steps) / self.epochs
            self.progress(
                0.9 * done, f"epoch {self.epoch + 1}/{self.epochs}, step {self.step}/{self.steps}{self.losses}"
            )

    def epoch_done(self, epoch: int, logs: dict[str, float]) -> None:
        """Record the epoch's losses and report them."""
        for key in self.history:
            self.history[key].append(float(logs.get(key, logs.get("lr", float("nan")))))
        self.epoch, self.step = epoch + 1, 0
        losses = f"loss {self.history['loss'][-1]:.4f}, val_loss {self.history['val_loss'][-1]:.4f}"
        self.losses = f" (epoch {epoch + 1}: {losses})"
        self.progress(0.9 * self.epoch / self.epochs, f"epoch {epoch + 1}/{self.epochs}: {losses}")


def validation_quality(star_dist: StarDist2D, split: TrainingSplit) -> list[dict[str, Any]]:
    """Return the notebook's QC row (section 5.2 metrics) for every validation image, with the optimised thresholds."""
    from labconstrictor_tools import check_cancel

    rows = []
    for index, (image, truth) in enumerate(zip(split.x_val, split.y_val, strict=True), start=1):
        check_cancel()
        labels, _ = star_dist.predict_instances(
            image, n_tiles=star_dist._guess_n_tiles(image), show_tile_progress=False
        )
        rows.append(metrics.qc_row(f"validation image {index}", truth, labels))
    return rows


def _training_tables(
    result: train.TrainingResult, validation: list[dict[str, Any]], target: Path, note: str
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Return the loss table, the validation QC table and the summary (naming the model folder) of a training."""
    import pandas as pd

    history = result.history
    rates = history.get("learning_rate", history.get("lr", [float("nan")] * len(history["loss"])))
    curve = pd.DataFrame(
        {
            "epoch": range(1, len(history["loss"]) + 1),
            "loss": history["loss"],
            "val_loss": history["val_loss"],
            "learning_rate": rates,
        }
    )
    quality = pd.DataFrame(validation, columns=list(metrics.QC_COLUMNS))
    hours, minutes, seconds = train.elapsed(result.seconds)
    f1 = float(quality["f1 score"].mean())
    summary = (
        f"Model **{target.name}** trained for {len(curve)} epochs in {hours} h {minutes} min {seconds} s ({note}). "
        f"Validation F1 **{f1:.3f}** (IoU 0.5). Use it with Model = 'Custom model folder' and model folder = {target}"
    )
    if f1 < LOW_F1:  # a model trained from scratch for a few epochs finds almost nothing (tester report, 1.1.1)
        summary += (
            ". The validation F1 is low: train for more epochs (often 100 or more from scratch), or start from "
            "2D_versatile_fluo, which needs only a few"
        )
    return curve, quality, summary


def quality_control_folder(
    model_folder: Path, images: Path, masks: Path, iou_threshold: float
) -> tuple[pd.DataFrame, str]:
    """Run the notebook's section 5 quality control of a model folder on test images; return the table and a summary.

    Writes what the notebook writes: ``Quality Control/Prediction/<image>.tif`` and
    ``Quality Control/Quality_Control for <model>.csv`` in the model folder.
    """
    import pandas as pd

    from stardist2d import bridge

    folder = Path(model_folder)
    star_dist = bridge.load(bridge.ModelChoice(model.CUSTOM_MODEL, folder))
    try:
        result = qc.run_quality_control(star_dist, folder, Path(images), Path(masks), iou_threshold, tool_hooks())
    except (ValueError, FileNotFoundError) as problem:
        raise tool_error("invalid_parameter", str(problem)) from None
    table = pd.DataFrame(result.rows, columns=list(metrics.QC_COLUMNS))
    summary = (
        f"{len(table)} images: mean F1 **{table['f1 score'].mean():.3f}**, mean IoU "
        f"{table['Prediction v. GT Intersection over Union'].mean():.3f}. "
        f"Table and predictions in {result.table.parent}"
    )
    return table, summary
