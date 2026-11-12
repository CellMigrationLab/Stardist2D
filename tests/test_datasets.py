"""The Zenodo dataset catalogue of section 3.1 (downloads are replaced by a small zip with the same layout)."""

import io
import zipfile
from unittest import mock

import numpy as np
import pytest
import tifffile

from stardist2d import data, datasets


def test_catalogue_entries_are_unique_and_complete():
    assert len(datasets.CATALOGUE) == 9
    assert len({d.record for d in datasets.CATALOGUE}) == 9
    for entry in datasets.CATALOGUE:
        assert entry.url.startswith(f"https://zenodo.org/records/{entry.record}/files/")
        assert all(
            getattr(entry, part) for part in ("training_images", "training_masks", "qc_images", "qc_masks", "model")
        )
    with pytest.raises(ValueError, match="unknown dataset"):
        datasets.by_title("nope")


def _zip_like(entry):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for folder in (entry.training_images, entry.training_masks, entry.qc_images, entry.qc_masks):
            image = io.BytesIO()
            tifffile.imwrite(image, np.zeros((8, 8), np.uint8))
            archive.writestr(f"{folder}/a.tif", image.getvalue())
        archive.writestr(f"{entry.model}/config.json", "{}")
        archive.writestr("__MACOSX/junk", "x")
    return buffer.getvalue()


class _Response:
    def __init__(self, content):
        self.content, self.status_code, self.headers = content, 200, {"Content-Length": str(len(content))}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def iter_content(self, size):
        yield self.content


@pytest.mark.parametrize("entry", datasets.CATALOGUE, ids=lambda d: d.record)
def test_download_unpacks_to_the_catalogued_folders(tmp_path, entry):
    reports = []
    with mock.patch("requests.get", return_value=_Response(_zip_like(entry))) as get:
        folder = datasets.download(entry, tmp_path, progress=lambda f, t: reports.append(f))
        assert datasets.download(entry, tmp_path) == folder  # already there: not downloaded again
    assert get.call_count == 1 and reports[-1] == 1.0
    assert data.training_pairs(folder / entry.training_images, folder / entry.training_masks)
    assert data.training_pairs(folder / entry.qc_images, folder / entry.qc_masks)
    assert (folder / entry.model / "config.json").is_file()
    assert not (folder / "__MACOSX").exists() and not (folder / entry.zip_name).exists()
