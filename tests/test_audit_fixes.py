"""Fixes from the independent audit of 1.0.2 (release blockers and quick fixes), on small synthetic data."""

import io
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from stardist2d import data, datasets, metrics, predict, prepare, pretrained, qc

ROOT = Path(__file__).resolve().parents[1]


# --- 1. outputs never overwrite or mix with the input images
@pytest.mark.parametrize("layout", ["same", "inside", "parent"])
def test_overlapping_results_and_image_folders_are_refused(tmp_path, layout):
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.tif").write_bytes(b"original")
    results = {"same": images, "inside": images / "results", "parent": tmp_path}[layout]
    with pytest.raises(ValueError, match="overlaps the image folder"):
        predict.predict_folder(None, images, results)
    assert (images / "a.tif").read_bytes() == b"original"


def test_a_symbolic_link_to_the_image_folder_is_refused(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    (tmp_path / "link").symlink_to(images, target_is_directory=True)
    with pytest.raises(ValueError, match="overlaps"):
        data.check_separate_folders(images, tmp_path / "link")
    data.check_separate_folders(images, tmp_path / "results")  # a separate folder is fine


def test_stack_and_big_image_folders_refuse_overlap(tmp_path):
    from stardist2d import batch

    images = tmp_path / "images"
    images.mkdir()
    with pytest.raises(ValueError, match="overlaps"):
        batch.predict_stacks(None, images, images)
    with pytest.raises(ValueError, match="overlaps"):
        batch.predict_big_folder(None, images, images / "out", batch.BigImageSettings())


# --- 4 and 15. normalisation and tiling settings are checked
@pytest.mark.parametrize("low, high", [(99.8, 1), (50, 50), (-1, 99), (1, 101)])
def test_bad_percentiles_are_refused(low, high):
    with pytest.raises(ValueError, match="0 <= low < high <= 100"):
        predict.Detection(low_percentile=low, high_percentile=high)


def test_a_misspelt_tiling_value_is_refused():
    with pytest.raises(ValueError, match="n_tiles must be"):
        predict.Detection(n_tiles="atuo")
    with pytest.raises(ValueError, match="prob_thresh"):
        predict.Detection(prob_thresh=1.5)


def test_normalisation_refuses_nan_and_a_zero_width_range_but_not_a_uniform_image():
    image = np.zeros((64, 64), np.float32)
    np.testing.assert_array_equal(prepare.normalize_image(image, (0, 1)), 0)  # uniform: nothing to segment
    sparse = image.copy()
    sparse[0, :5] = 1000  # 5 bright pixels: the 1 and 99.8 percentiles are both 0
    with pytest.raises(ValueError, match="percentiles of the image are equal"):
        prepare.normalize_image(sparse, (0, 1))
    image[3, 3] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        prepare.normalize_image(image, (0, 1))


# --- 3. the tools' model cache notices new weights and thresholds
def test_the_model_cache_key_follows_weights_and_thresholds(tmp_path):
    from stardist2d import bridge

    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "weights_best.h5").write_bytes(b"a")
    before = bridge._folder_state(str(tmp_path))
    (tmp_path / "weights_best.h5").write_bytes(b"bb")
    after_weights = bridge._folder_state(str(tmp_path))
    (tmp_path / "thresholds.json").write_text("{}")
    assert len({before, after_weights, bridge._folder_state(str(tmp_path))}) == 3
    assert bridge._folder_state("") == ()


# --- 5. an undefined QC value comes with its reason
def test_an_undefined_iou_has_a_reason_beside_the_table(tmp_path):
    empty = np.zeros((16, 16), np.uint16)
    row = metrics.qc_row("blank.tif", empty, empty)
    assert np.isnan(row[metrics.IOU_COLUMN])  # kept as 1.20.8
    assert metrics.undefined_values(row) == [(metrics.IOU_COLUMN, metrics.BOTH_MASKS_EMPTY)]
    table = tmp_path / "Quality_Control for m.csv"
    path = qc._write_undefined_values(table, [row], predict.Hooks())
    assert path.read_text() == f"image,column,reason\nblank.tif,{metrics.IOU_COLUMN},both_masks_empty\n"
    assert qc._write_undefined_values(table, [], predict.Hooks()) is None


# --- 6. quality control refuses what a 16-bit label image cannot hold
def test_quality_control_refuses_more_than_65535_objects(monkeypatch):
    from labconstrictor_tools import ToolError

    from stardist2d import bridge

    big = np.arange(70000, dtype=np.int32).reshape(280, 250)
    monkeypatch.setattr(bridge, "load", lambda choice: None)
    monkeypatch.setattr(bridge, "check_image", lambda image, star_dist: image)
    monkeypatch.setattr(predict, "predict_image", lambda *args, **kwargs: (big, {}))
    with pytest.raises(ToolError, match="16-bit"):
        bridge.quality_control(np.zeros((280, 250)), big, bridge.ModelChoice("2D_versatile_fluo"), 0.5)


# --- 7. an interrupted dataset download is never trusted
class _Response:
    def __init__(self, content, announced=None):
        self.content, self.status_code = content, 200
        self.headers = {"Content-Length": str(announced or len(content))}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def iter_content(self, size):
        yield self.content


def _zip(entry, parts):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for part in parts:
            archive.writestr(f"{getattr(entry, part)}/a.tif" if part != "model" else f"{entry.model}/config.json", "x")
    return buffer.getvalue()


ALL_PARTS = ("training_images", "training_masks", "qc_images", "qc_masks", "model")


def test_a_folder_left_by_an_interrupted_download_is_replaced(tmp_path):
    entry = datasets.CATALOGUE[0]
    (tmp_path / entry.record / entry.training_images).mkdir(parents=True)  # what an interrupted unzip leaves
    with mock.patch("requests.get", return_value=_Response(_zip(entry, ALL_PARTS))) as get:
        folder = datasets.download(entry, tmp_path)
    assert get.call_count == 1 and (folder / datasets.COMPLETE_MARKER).is_file()
    assert (folder / entry.qc_masks / "a.tif").is_file()


def test_an_archive_missing_a_part_or_cut_short_is_refused(tmp_path):
    entry = datasets.CATALOGUE[0]
    with mock.patch("requests.get", return_value=_Response(_zip(entry, ALL_PARTS[:-1]))):
        with pytest.raises(RuntimeError, match="lacks"):
            datasets.download(entry, tmp_path)
    content = _zip(entry, ALL_PARTS)
    with mock.patch("requests.get", return_value=_Response(content, announced=len(content) + 10)):
        with pytest.raises(RuntimeError, match="stopped after"):
            datasets.download(entry, tmp_path)
    assert not (tmp_path / entry.record).exists()


# --- 16. pretrained-model downloads are complete and contain a model
def test_a_short_model_download_leaves_nothing(tmp_path):
    with mock.patch("requests.get", return_value=_Response(b"abc", announced=10)):
        with pytest.raises(RuntimeError, match="stopped after"):
            pretrained._download("https://example.org/m.zip", tmp_path / "m.zip", 5)
    assert list(tmp_path.iterdir()) == []


def test_an_archive_without_a_model_is_refused(tmp_path):
    archive = tmp_path / "m.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("readme.txt", "no model")
    with pytest.raises(RuntimeError, match="does not contain a StarDist model"):
        pretrained._unpack(archive, tmp_path / "model")
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("config.json", "{}")
        handle.writestr("weights_best.h5", "w")
    (tmp_path / "model").mkdir()
    pretrained._unpack(archive, tmp_path / "model")
    assert sorted(p.name for p in (tmp_path / "model").iterdir()) == ["config.json", "weights_best.h5"]


# --- 10. Keras 3 already loaded is refused, whatever the environment variable says
@pytest.mark.slow
def test_keras_3_loaded_before_stardist2d_is_refused():
    code = (
        "import tensorflow as tf; _ = tf.keras.layers\n"
        "import stardist2d\n"
        "from stardist2d.keras_backend import check_keras_backend\n"
        "check_keras_backend()\n"
    )
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "TF_CPP_MIN_LOG_LEVEL": "3"}
    environment.pop("TF_USE_LEGACY_KERAS", None)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=environment, timeout=600)
    assert result.returncode != 0 and "already uses Keras 3" in result.stderr
