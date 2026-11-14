"""The rehosted lab and Model Zoo models (stardist2d.model_catalogue): files, checksums, registry, citations."""

import hashlib
import typing
import zipfile
from pathlib import Path

import pytest
import tifffile

import stardist2d_lc_tools
from stardist2d import model, model_catalogue, pretrained

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


@pytest.fixture(autouse=True)
def local_models(monkeypatch):
    """Download from this checkout's models/ folder: the release tag does not exist before the release."""
    monkeypatch.setenv(model_catalogue.URL_VARIABLE, MODELS.as_uri() + "/")


def test_every_model_zip_is_there_with_its_checksum_and_files():
    assert {p.stem for p in MODELS.glob("*.zip")} == {m.name for m in model_catalogue.CATALOGUE}
    for entry in model_catalogue.CATALOGUE:
        data = (MODELS / f"{entry.name}.zip").read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry.sha256, entry.name
        with zipfile.ZipFile(MODELS / f"{entry.name}.zip") as archive:
            assert sorted(archive.namelist()) == ["README.md", "config.json", "thresholds.json", "weights_best.h5"]
            assert entry.licence.lower() in archive.read("README.md").decode().lower()


def test_the_tools_list_exactly_the_catalogue():
    listed = set(typing.get_args(stardist2d_lc_tools.FluoModel))
    assert {m.label for m in model_catalogue.CATALOGUE} <= listed
    assert set(typing.get_args(stardist2d_lc_tools.CatalogueModels)) == {m.label for m in model_catalogue.CATALOGUE}


def test_labels_are_accepted_and_cited():
    for entry in model_catalogue.CATALOGUE:
        assert model.ModelChoice(entry.label).source == entry.label
        assert entry.cite in model_catalogue.citation(entry.label)
        assert entry.label in pretrained.PRETRAINED_CHOICES
    assert model_catalogue.citation("2D_versatile_fluo") == ""
    with pytest.raises(ValueError, match="CML: HUVEC nuclei"):
        model.ModelChoice("CML: nothing")


@pytest.mark.slow
def test_every_catalogue_model_loads_through_the_registry_and_predicts():
    image = tifffile.imread(ROOT / "example_data/StarDist_2D/Test - Images/test_01.tif")
    from csbdeep.utils import normalize

    found = {}
    for entry in model_catalogue.CATALOGUE:
        star_dist = model.load_model(model.ModelChoice(entry.label))
        assert (star_dist.config.n_channel_in, star_dist.config.n_rays) == (1, 32)
        labels, _ = star_dist.predict_instances(normalize(image, 1, 99.8), n_tiles=(2, 2), show_tile_progress=False)
        found[entry.name] = int(labels.max())
    # the example image is fluorescent nuclei: the fluorescence models find them
    assert all(found[m.name] > 100 for m in model_catalogue.CATALOGUE if m.name == "cml_huvec_nuclei")


@pytest.mark.slow
def test_section_3_4_copies_a_catalogue_model(tmp_path):
    folder = pretrained.fetch_model_folder("CML: monocytes (brightfield)", tmp_path)
    assert folder.name == "cml_monocytes_bf"
    assert pretrained.weights_file("CML: monocytes (brightfield)", folder, "last").name == "weights_best.h5"


@pytest.mark.slow
def test_a_damaged_download_is_refused(tmp_path, monkeypatch):
    bad = tmp_path / "models"
    bad.mkdir()
    entry = model_catalogue.CATALOGUE[0]
    (bad / f"{entry.name}.zip").write_bytes(b"not the model")
    monkeypatch.setenv(model_catalogue.URL_VARIABLE, bad.as_uri() + "/")
    monkeypatch.setenv("HOME", str(tmp_path))  # Keras caches in ~/.keras: an empty cache, so the file is downloaded
    with pytest.raises(RuntimeError, match="could not be downloaded, or the download failed its checksum"):
        model.load_model(model.ModelChoice(entry.label))
    assert not (tmp_path / ".keras" / "models" / "StarDist2D" / entry.name / "config.json").exists()
