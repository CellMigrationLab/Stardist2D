"""Command line: ``python -m stardist2d <step> ...`` for scripts, clusters and batch runs.

The same functions as the notebook and the Fiji / napari / QuPath tools, with the notebook's defaults.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from stardist2d import batch, config, model, predict, provenance, qc, train
from stardist2d import environment as env


def _model_choice(arguments: argparse.Namespace) -> model.ModelChoice:
    """Return the model named on the command line (a pretrained name or a folder)."""
    if arguments.model in model.PRETRAINED_MODELS:
        return model.ModelChoice(arguments.model)
    return model.ModelChoice(model.CUSTOM_MODEL, Path(arguments.model))


def _detection(arguments: argparse.Namespace) -> predict.Detection:
    """Return the detection settings given on the command line."""
    return predict.Detection(arguments.prob_thresh, arguments.nms_thresh, n_tiles="auto" if arguments.tiles else None)


def _report(fraction: float, message: str) -> None:
    """Print a progress line."""
    print(f"[{fraction:4.0%}] {message}", flush=True)


def run_train(arguments: argparse.Namespace) -> None:
    """Train a model (sections 3 and 4)."""
    settings = config.TrainingSettings(
        number_of_epochs=arguments.epochs,
        use_default_advanced_parameters=not arguments.advanced,
        batch_size=arguments.batch_size,
        number_of_steps=arguments.steps,
        patch_size=arguments.patch_size,
    )
    augmentation = config.AugmentationSettings(arguments.augment > 1, max(arguments.augment, 1))
    data = train.load_training_data(arguments.images, arguments.masks, settings, augmentation)
    folder = arguments.model_folder
    result = train.train(
        train.new_model(data, folder, arguments.overwrite),
        data,
        settings.number_of_epochs,
        augmentation.use_data_augmentation,
        lambda epoch, logs: _report(0, f"epoch {epoch + 1}: {logs}"),
    )
    train.write_training_evaluation(folder, result.history)
    provenance.write_run_file(
        folder,
        "train",
        {"training": data.settings, "augmentation": augmentation},
        sorted(arguments.images.glob("*.tif")),
        env.collect(),
    )
    print(f"Model written to {folder} in {train.elapsed(result.seconds)} (h, min, s)")


def run_qc(arguments: argparse.Namespace) -> None:
    """Evaluate a model against ground truth (section 5.2)."""
    star_dist = model.load_model_folder(arguments.model_folder)
    result = qc.run_quality_control(
        star_dist, arguments.model_folder, arguments.images, arguments.masks, hooks=predict.Hooks(progress=_report)
    )
    print(f"Quality control table: {result.table}")


def run_predict(arguments: argparse.Namespace) -> None:
    """Predict a folder of images, stacks or very large images (sections 6.1 and 6.2)."""
    star_dist = model.load_model(_model_choice(arguments))
    hooks = predict.Hooks(progress=_report)
    if arguments.kind == "images":
        predict.predict_folder(
            star_dist, arguments.images, arguments.results, _detection(arguments), not arguments.no_rois, hooks
        )
    elif arguments.kind == "stacks":
        batch.predict_stacks(star_dist, arguments.images, arguments.results, not arguments.no_rois, hooks)
    else:
        settings = batch.BigImageSettings(region_of_interests=not arguments.no_rois, is_stack=arguments.big_stacks)
        batch.predict_big_folder(star_dist, arguments.images, arguments.results, settings, hooks)
    provenance.write_run_file(
        arguments.results,
        f"predict {arguments.kind}",
        {"model": arguments.model},
        sorted(arguments.images.glob("*.tif")),
        env.collect(),
    )


def parser() -> argparse.ArgumentParser:
    """Return the command line parser."""
    main = argparse.ArgumentParser(prog="python -m stardist2d", description=__doc__)
    steps = main.add_subparsers(dest="step", required=True)
    training = steps.add_parser("train", help="train a model (notebook sections 3-4)")
    training.add_argument("images", type=Path)
    training.add_argument("masks", type=Path)
    training.add_argument("model_folder", type=Path)
    training.add_argument("--epochs", type=int, default=100)
    training.add_argument("--augment", type=int, default=4, help="multiply the dataset by (1 = no augmentation)")
    training.add_argument("--advanced", action="store_true", help="use the batch size, steps and patch size below")
    training.add_argument("--batch-size", type=int, default=2)
    training.add_argument("--steps", type=int, default=50)
    training.add_argument("--patch-size", type=int, default=512)
    training.add_argument("--overwrite", action="store_true", help="delete an existing model folder of that name")
    training.set_defaults(run=run_train)
    quality = steps.add_parser("qc", help="quality control against ground truth (section 5.2)")
    quality.add_argument("model_folder", type=Path)
    quality.add_argument("images", type=Path)
    quality.add_argument("masks", type=Path)
    quality.set_defaults(run=run_qc)
    prediction = steps.add_parser("predict", help="predict a folder (sections 6.1-6.2)")
    prediction.add_argument("model", help=f"a model folder or one of {model.PRETRAINED_MODELS}")
    prediction.add_argument("images", type=Path)
    prediction.add_argument("results", type=Path)
    prediction.add_argument("--kind", choices=("images", "stacks", "big"), default="images")
    prediction.add_argument("--big-stacks", action="store_true", help="with --kind big: each file is a stack")
    prediction.add_argument("--no-rois", action="store_true", help="do not write ImageJ ROI zips")
    prediction.add_argument("--prob-thresh", type=float)
    prediction.add_argument("--nms-thresh", type=float)
    prediction.add_argument("--tiles", action="store_true", help="predict in tiles chosen from the image size")
    prediction.set_defaults(run=run_predict)
    return main


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line and return the exit code (1 with a one-line message for expected problems)."""
    arguments = parser().parse_args(argv)
    try:
        arguments.run(arguments)
    except (ValueError, FileNotFoundError, FileExistsError) as problem:
        print(f"stardist2d: {problem}", file=sys.stderr)
        return 1
    return 0
