import numpy as np
import pytest
import tifffile

from stardist2d import data


def _write(folder, names):
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        tifffile.imwrite(folder / name, np.zeros((8, 8), np.uint8))


def test_pairs_are_matched_by_name_in_sorted_order(tmp_path):
    _write(tmp_path / "images", ["b.tif", "a.tif"])
    _write(tmp_path / "masks", ["a.tif", "b.tif"])
    pairs = data.training_pairs(tmp_path / "images", tmp_path / "masks")
    assert [(image.name, mask.name) for image, mask in pairs] == [("a.tif", "a.tif"), ("b.tif", "b.tif")]


def test_an_image_without_mask_is_refused_not_ignored(tmp_path):
    # 1.20.8 zipped the two sorted lists and ignored the extra image.
    _write(tmp_path / "images", ["a.tif", "b.tif", "c.tif"])
    _write(tmp_path / "masks", ["a.tif", "b.tif"])
    with pytest.raises(ValueError, match=r"without a mask: \['c.tif'\]"):
        data.training_pairs(tmp_path / "images", tmp_path / "masks")


def test_only_tif_files_are_read(tmp_path):
    _write(tmp_path / "images", ["a.tif"])
    (tmp_path / "images" / "notes.txt").write_text("x")
    assert [path.name for path in data.image_files(tmp_path / "images")] == ["a.tif"]


@pytest.mark.parametrize("folder", ["", "."])
def test_an_empty_path_is_refused(folder):
    from pathlib import Path

    with pytest.raises(ValueError, match="no image folder"):
        data.image_files(Path(folder))


def test_a_missing_or_empty_folder_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        data.image_files(tmp_path / "missing")
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no .tif file"):
        data.image_files(tmp_path / "empty")
