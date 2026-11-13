"""Stacks and big images: the outputs 1.20.8 names, with the documented fixes."""

import shutil
import zipfile

import numpy as np
import pytest
import tifffile

from stardist2d import batch, predict
from stardist2d import model as model_module

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    from stardist.models import StarDist2D

    target = tmp_path_factory.mktemp("models") / "versatile"
    shutil.copytree(
        StarDist2D.from_pretrained("2D_versatile_fluo").logdir, target, ignore=shutil.ignore_patterns("*.zip")
    )
    return model_module.load_model_folder(target)


@pytest.fixture(scope="module")
def nuclei():
    from stardist.data import test_image_nuclei_2d

    return test_image_nuclei_2d()[:192, :192]


def test_stack_frames_are_normalised_positive_and_labels_are_integers(tmp_path, model, nuclei):
    images = tmp_path / "stacks"
    images.mkdir()
    tifffile.imwrite(images / "movie.tif", np.stack([nuclei, np.roll(nuclei, 5, axis=1), nuclei]))
    result = batch.predict_stacks(model, images, tmp_path / "out")
    frames = sorted((tmp_path / "out" / "movie").glob("frame_*.tif"))
    assert [f.name for f in frames] == ["frame_00000.tif", "frame_00001.tif", "frame_00002.tif"]
    labels = tifffile.imread(frames[0])
    assert np.issubdtype(labels.dtype, np.integer)
    assert labels.max() > 5
    counts = (tmp_path / "out" / "movie_Nuclei_number.csv").read_text().splitlines()
    assert counts[0].startswith("0,") and len(counts) == 3
    with zipfile.ZipFile(tmp_path / "out" / "movie.zip") as rois:
        assert {name.split("_")[0] for name in rois.namelist()} == {"001", "002", "003"}  # one position per frame
    assert set(result.counts) == {"movie.tif"}


def test_a_finished_stack_is_skipped_and_a_partial_one_resumed(tmp_path, model, nuclei):
    images = tmp_path / "stacks"
    images.mkdir()
    tifffile.imwrite(images / "movie.tif", np.stack([nuclei, nuclei]))
    batch.predict_stacks(model, images, tmp_path / "out")
    assert batch.predict_stacks(model, images, tmp_path / "out").skipped == ("movie.tif",)
    clean = (tmp_path / "out" / "movie_Nuclei_number.csv").read_text()
    (tmp_path / "out" / "movie" / "frame_00001.tif").unlink()
    resumed = batch.predict_stacks(model, images, tmp_path / "out")
    # The counts table stays complete: frame 0's count comes from its label image, written before.
    assert [index for index, _ in resumed.counts["movie.tif"]] == [0, 1]
    assert (tmp_path / "out" / "movie_Nuclei_number.csv").read_text() == clean


def test_resume_ignores_stray_files_and_redoes_a_corrupt_frame(tmp_path, model, nuclei):
    """1.20.8 counted every file of the folder; a stray file or a half-written frame must not count as done."""
    images = tmp_path / "stacks"
    images.mkdir()
    tifffile.imwrite(images / "movie.tif", np.stack([nuclei, nuclei, nuclei]))
    batch.predict_stacks(model, images, tmp_path / "out")
    clean = (tmp_path / "out" / "movie_Nuclei_number.csv").read_text()
    frames = tmp_path / "out" / "movie"
    (frames / "frame_00002.tif").unlink()
    (frames / "frame_00001.tif").write_bytes(b"half a tiff")
    (frames / "notes.txt").write_text("stray")
    assert batch.completed_frames(frames, 3) == 1
    resumed = batch.predict_stacks(model, images, tmp_path / "out")
    assert [index for index, _ in resumed.counts["movie.tif"]] == [0, 1, 2]
    assert (tmp_path / "out" / "movie_Nuclei_number.csv").read_text() == clean


def test_normalise_frame_fixes_the_inverted_division():
    frame = np.array([[0.0, 50.0, 100.0]])
    scaled = predict.normalise_frame(frame, np.array(0.0), np.array(100.0))
    np.testing.assert_allclose(scaled, [[0.0, 0.5, 1.0]], atol=1e-6)


def test_big_image_range_uses_percentiles_not_0_255():
    image = np.linspace(0, 60000, 256 * 256).reshape(256, 256).astype(np.uint16)
    low, high = predict.big_image_range(image, 1, 99.8)
    assert low > 0 and 50000 < high <= 60000  # 1.20.8 would have used 0 and 255


def test_big_image_outputs(tmp_path, model, nuclei):
    images = tmp_path / "big"
    images.mkdir()
    tifffile.imwrite(images / "slide.tif", np.tile(nuclei, (4, 4)))
    settings = batch.BigImageSettings(is_stack=False)  # 1.20.8 defaults: blocks of 512, overlap 50
    assert batch.predict_big_folder(model, images, tmp_path / "out", settings) == ["slide.tif"]
    labels = tifffile.imread(tmp_path / "out" / "slide.tif")
    assert labels.shape == (768, 768) and labels.max() > 20
    assert (tmp_path / "out" / "slidelabels_roi.zip").is_file()


def test_big_image_stacks_and_folders_of_several_images(tmp_path, model, nuclei):
    """Every frame of a stack and every image of a folder gets its own temporary array (1.1.1 failed on the 2nd)."""
    images = tmp_path / "big"
    images.mkdir()
    tifffile.imwrite(images / "movie.tif", np.stack([np.tile(nuclei, (4, 4))] * 2))
    assert batch.predict_big_folder(model, images, tmp_path / "out", batch.BigImageSettings()) == ["movie.tif"]
    frames = sorted((tmp_path / "out" / "movie").iterdir())
    assert [f.name for f in frames] == ["frame_00000.tif", "frame_00001.tif"]
    assert all(tifffile.imread(f).max() > 10 for f in frames)
    singles = tmp_path / "singles"
    singles.mkdir()
    for name in ("a.tif", "b.tif"):
        tifffile.imwrite(singles / name, np.tile(nuclei, (4, 4)))
    settings = batch.BigImageSettings(is_stack=False)
    assert batch.predict_big_folder(model, singles, tmp_path / "out2", settings) == ["a.tif", "b.tif"]


def test_a_big_image_smaller_than_a_block_is_refused(tmp_path, model, nuclei):
    images = tmp_path / "small"
    images.mkdir()
    tifffile.imwrite(images / "small.tif", nuclei)
    with pytest.raises(ValueError, match="smaller than block_size 512"):
        batch.predict_big_folder(model, images, tmp_path / "out", batch.BigImageSettings(is_stack=False))


def test_big_image_settings_are_checked():
    with pytest.raises(ValueError, match="twice min_overlap"):
        batch.BigImageSettings(block_size=64, min_overlap=40)
    with pytest.raises(ValueError, match="at least one output"):
        batch.BigImageSettings(mask_images=False, region_of_interests=False)
