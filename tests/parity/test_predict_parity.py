"""Prediction of a folder writes what cell 41 of 1.20.8 writes (labels, ROIs, centres, counts).

Uses StarDist's pretrained 2D_versatile_fluo (downloaded once into ~/.keras) and its test image.
"""

import os
import shutil
import zipfile

import numpy as np
import pandas as pd
import pytest
import tifffile

from reference_notebook import run
from stardist2d import model as model_module
from stardist2d import predict

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def pretrained_folder(tmp_path_factory):
    from stardist.models import StarDist2D

    loaded = StarDist2D.from_pretrained("2D_versatile_fluo")
    source = loaded.logdir
    target = tmp_path_factory.mktemp("models") / "versatile"
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("*.zip"))
    return target


@pytest.fixture(scope="module")
def images(tmp_path_factory):
    from stardist.data import test_image_nuclei_2d

    folder = tmp_path_factory.mktemp("images")
    full = test_image_nuclei_2d()
    tifffile.imwrite(folder / "first.tif", full[:256, :256])
    tifffile.imwrite(folder / "second.tif", full[256:, 128:448])
    return folder


def _reference(tmp_path, images, pretrained_folder):
    from csbdeep.utils import normalize
    from stardist import export_imagej_rois, random_label_cmap
    from stardist.models import StarDist2D

    results = tmp_path / "reference"
    namespace = {
        "Data_type": 1,
        "Data_folder": str(images),
        "Results_folder": str(results),
        "np": np,
        "os": os,
        "pd": pd,
        "glob": __import__("glob").glob,
        "imread": tifffile.imread,
        "imwrite": tifffile.imwrite,
        "normalize": normalize,
        "StarDist2D": StarDist2D,
        "export_imagej_rois": export_imagej_rois,
        "random_label_cmap": random_label_cmap,
        "Prediction_model_name": pretrained_folder.name,
        "Prediction_model_path": str(pretrained_folder.parent),
        "sys": __import__("sys"),
        "print": lambda *args, **kwargs: None,
        "plt": _NoPlot(),
    }
    results.mkdir()
    cwd = os.getcwd()
    try:
        run(41, lambda code: code.startswith("if Data_type == 1"), namespace)
    finally:
        os.chdir(cwd)
    return results


class _NoPlot:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def test_single_image_outputs_match_1_20_8(tmp_path, images, pretrained_folder):
    reference = _reference(tmp_path, images, pretrained_folder)
    model = model_module.load_model_folder(pretrained_folder)
    ours = tmp_path / "ours"
    result = predict.predict_folder(model, images, ours)

    for name in ("first", "second"):
        np.testing.assert_array_equal(tifffile.imread(ours / f"{name}.tif"), tifffile.imread(reference / f"{name}.tif"))
        assert (ours / f"{name}_Nuclei_centre.csv").read_text() == (reference / f"{name}_Nuclei_centre.csv").read_text()
        with zipfile.ZipFile(ours / f"{name}.zip") as a, zipfile.ZipFile(reference / f"{name}.zip") as b:
            assert a.namelist() == b.namelist()
            assert all(a.read(entry) == b.read(entry) for entry in a.namelist())
    assert (ours / "Nuclei_count.csv").read_text() == (reference / "Nuclei_count.csv").read_text()
    assert result.counts[0][1] > 10


def test_progress_and_cancel_hooks_are_called(images, pretrained_folder):
    model = model_module.load_model_folder(pretrained_folder)
    image = tifffile.imread(images / "first.tif")
    reports = []
    hooks = predict.Hooks(progress=lambda fraction, message: reports.append((fraction, message)))
    predict.predict_image(model, image, predict.Detection(n_tiles=(2, 2)), hooks)
    assert reports[-1][0] == 1.0
    assert [message for _, message in reports if message.startswith("predicted tile")] == [
        f"predicted tile {done} of 4" for done in range(1, 5)
    ]

    class Stop(Exception):
        pass

    def cancel():
        raise Stop

    with pytest.raises(Stop):
        predict.predict_image(model, image, predict.Detection(n_tiles=(2, 2)), predict.Hooks(check_cancel=cancel))


def test_rois_are_optional(tmp_path, images, pretrained_folder):
    model = model_module.load_model_folder(pretrained_folder)
    predict.predict_folder(model, images, tmp_path / "out", export_rois=False)
    assert not list((tmp_path / "out").glob("*.zip"))
