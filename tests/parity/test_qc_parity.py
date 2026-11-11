"""The QC table equals the one cell 36 of 1.20.8 writes, byte for byte."""

import csv
import os

import numpy as np
import pytest
import tifffile

from reference_notebook import run
from stardist2d import export, metrics


def _labels(rng: np.random.Generator, n: int, shape: tuple[int, int] = (64, 64)) -> np.ndarray:
    labels = np.zeros(shape, np.uint16)
    for label in range(1, n + 1):
        y, x = rng.integers(4, shape[0] - 12), rng.integers(4, shape[1] - 12)
        labels[y : y + rng.integers(4, 10), x : x + rng.integers(4, 10)] = label
    return labels


def _dataset(tmp_path):
    rng = np.random.default_rng(1)
    names = ["a.tif", "b.tif", "c.tif"]
    folders = {kind: tmp_path / kind for kind in ("images", "masks", "model/qc/Quality Control/Prediction")}
    for folder in folders.values():
        folder.mkdir(parents=True)
    pairs = {}
    for index, name in enumerate(names):
        truth = _labels(rng, 6 + index)
        prediction = truth.copy() if index == 0 else _labels(rng, 5 + index)
        tifffile.imwrite(folders["images"] / name, rng.random((64, 64)).astype(np.float32))
        tifffile.imwrite(folders["masks"] / name, truth)
        tifffile.imwrite(folders["model/qc/Quality Control/Prediction"] / name, prediction)
        pairs[name] = (truth, prediction)
    return folders, pairs


def test_qc_table_matches_1_20_8(tmp_path):
    from stardist.matching import matching

    folders, pairs = _dataset(tmp_path)
    namespace = {
        "QC_model_path": str(tmp_path / "model"), "QC_model_name": "qc", "Source_QC_folder": str(folders["images"]),
        "Target_QC_folder": str(folders["masks"]), "imread": tifffile.imread, "matching": matching, "np": np,
        "csv": csv, "os": os, "print": lambda *args, **kwargs: None,
    }
    run(36, lambda code: code.startswith("with open("), namespace)
    reference = tmp_path / "model/qc/Quality Control/Quality_Control for qc.csv"

    # 1.20.8 lists the images in os.listdir order; the package sorts them. Compare row by row.
    reference_rows = {row[0]: row for row in csv.reader(reference.read_text().splitlines())}
    rows = [metrics.qc_row(name, *pairs[name]) for name in sorted(pairs)]
    ours = export.write_qc_table(tmp_path / "ours.csv", rows)
    our_rows = {row[0]: row for row in csv.reader(ours.read_text().splitlines())}
    assert our_rows == reference_rows


def test_identical_masks_give_perfect_scores():
    truth = _labels(np.random.default_rng(3), 5)
    row = metrics.qc_row("x.tif", truth, truth)
    assert row["Prediction v. GT Intersection over Union"] == 1.0
    assert row["f1 score"] == 1.0
    assert row["false positive"] == row["false negative"] == 0


def test_two_empty_masks_give_nan_iou_without_a_warning():
    empty = np.zeros((8, 8), np.uint16)
    with np.errstate(all="raise"):
        assert np.isnan(metrics.binary_iou(empty, empty))


def test_shapes_must_match():
    with pytest.raises(ValueError, match="differ in shape"):
        metrics.qc_row("x.tif", np.zeros((4, 4), int), np.zeros((4, 5), int))
