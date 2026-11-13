"""Training and folder quality control as the Fiji / napari / QuPath tools run them (stardist2d.bridge_train)."""

import json
import shutil

import labconstrictor_tools
import pytest
from labconstrictor_tools import Cancelled, ToolError

from stardist2d import bridge_train, config


def _settings(epochs=1):
    return config.TrainingSettings(
        number_of_epochs=epochs,
        use_default_advanced_parameters=False,
        batch_size=2,
        number_of_steps=2,
        patch_size=64,
        percentage_validation=25,
        n_rays=16,
        grid_parameter=2,
    )


def _request(images, masks, parent, name="tiny", **changes):
    return bridge_train.TrainingRequest(images, masks, parent, name, _settings(), **changes)


@pytest.mark.parametrize("name", ["", "../up", "a/b", "a\\b", ".hidden", "x" * 101])
def test_a_bad_model_name_is_refused(tmp_path, name):
    with pytest.raises(ToolError, match="model name"):
        bridge_train.model_target(_request(tmp_path, tmp_path, tmp_path, name))


def test_an_existing_model_folder_needs_overwrite(tmp_path):
    (tmp_path / "models" / "tiny").mkdir(parents=True)
    request = _request(tmp_path / "i", tmp_path / "m", tmp_path / "models")
    with pytest.raises(ToolError, match="already exists"):
        bridge_train.model_target(request)
    assert bridge_train.model_target(_request(tmp_path / "i", tmp_path / "m", tmp_path / "models", overwrite=True))


def test_the_model_folder_must_not_be_or_contain_a_training_folder(tmp_path):
    images = tmp_path / "data" / "images"
    with pytest.raises(ToolError, match="training folders"):
        bridge_train.model_target(_request(images, tmp_path / "masks", tmp_path, "data"))


def test_starting_from_an_own_model_needs_its_folder(tmp_path):
    with pytest.raises(ToolError, match="choose the model folder"):
        bridge_train.start_weights(_request(tmp_path, tmp_path, tmp_path, start_from="Own model folder"))
    with pytest.raises(ToolError, match="start from must be one of"):
        bridge_train.start_weights(_request(tmp_path, tmp_path, tmp_path, start_from="mine"))
    assert bridge_train.start_weights(_request(tmp_path, tmp_path, tmp_path)) == (None, None, "trained from scratch")


@pytest.mark.slow
def test_training_writes_the_notebook_outputs_and_tables(tmp_path, training_folders):
    images, masks = training_folders
    curve, quality, summary = bridge_train.train_model(_request(images, masks, tmp_path))
    folder = tmp_path / "tiny"
    assert list(curve.columns) == ["epoch", "loss", "val_loss", "learning_rate"] and len(curve) == 1
    assert len(quality) == 1 and "f1 score" in quality.columns
    assert "**tiny**" in summary and str(folder) in summary
    if quality["f1 score"].mean() < bridge_train.LOW_F1:  # 1 epoch from scratch: the summary says what to do
        assert "train for more epochs" in summary and "2D_versatile_fluo" in summary
    for name in ("config.json", "thresholds.json", "weights_best.h5", "Quality Control/training_evaluation.csv"):
        assert (folder / name).is_file(), name
    run = json.loads((folder / "stardist2d_run.json").read_text())
    assert run["settings"]["state"] == "finished: 1 epochs, thresholds optimised"
    assert len(run["inputs"]) == 8


@pytest.mark.slow
def test_cancel_keeps_what_was_saved(tmp_path, training_folders, monkeypatch):
    """Cancel during epoch 2 stops at once; epoch 1's best weights, loss row and run file stay."""
    images, masks = training_folders
    calls = []

    def check_cancel():
        calls.append(1)
        if len(calls) == 3:  # 2 steps per epoch: the first batch of epoch 2
            raise Cancelled()

    monkeypatch.setattr(labconstrictor_tools, "check_cancel", check_cancel)
    request = bridge_train.TrainingRequest(images, masks, tmp_path, "tiny", _settings(epochs=2))
    with pytest.raises(Cancelled):
        bridge_train.train_model(request)
    folder = tmp_path / "tiny"
    assert (folder / "weights_best.h5").is_file() and not (folder / "thresholds.json").exists()
    rows = (folder / "Quality Control" / "training_evaluation.csv").read_text().splitlines()
    assert len(rows) == 2  # header and epoch 1
    run = json.loads((folder / "stardist2d_run.json").read_text())
    assert run["settings"]["state"] == "stopped after epoch 1 of 2; thresholds not optimised"


@pytest.mark.slow
def test_fine_tuning_from_an_own_model(tmp_path, training_folders, pretrained_folder):
    images, masks = training_folders
    settings = config.TrainingSettings(
        number_of_epochs=1, use_default_advanced_parameters=False, number_of_steps=1, patch_size=64
    )
    request = bridge_train.TrainingRequest(
        images, masks, tmp_path, "tuned", settings, start_from="Own model folder", start_folder=pretrained_folder
    )
    _, _, summary = bridge_train.train_model(request)
    assert "started from versatile" in summary


@pytest.mark.slow
def test_a_starting_model_with_other_rays_is_refused(tmp_path, training_folders, pretrained_folder):
    images, masks = training_folders
    request = _request(images, masks, tmp_path, start_from="Own model folder", start_folder=pretrained_folder)
    with pytest.raises(ToolError, match="does not fit these settings"):
        bridge_train.train_model(request)


@pytest.mark.slow
def test_quality_control_on_a_folder(tmp_path, training_folders, pretrained_folder):
    images, masks = training_folders
    model = tmp_path / "versatile"
    shutil.copytree(pretrained_folder, model)
    table, summary = bridge_train.quality_control_folder(model, images, masks, 0.5)
    assert len(table) == 4 and "mean F1" in summary
    assert (model / "Quality Control" / "Quality_Control for versatile.csv").is_file()
