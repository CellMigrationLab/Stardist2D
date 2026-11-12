"""Training, quality control and BioImage Model Zoo export on small synthetic data (TensorFlow, CPU)."""

import zipfile

import numpy as np
import pytest
import tifffile

from stardist2d import bioimage, config, qc, train
from stardist2d import model as model_module

pytestmark = pytest.mark.slow


def _settings():
    return config.TrainingSettings(
        number_of_epochs=1,
        use_default_advanced_parameters=False,
        batch_size=2,
        number_of_steps=2,
        patch_size=64,
        percentage_validation=25,
        n_rays=16,
        grid_parameter=2,
    )


def test_training_writes_a_loadable_model_and_its_evaluation(tmp_path, training_folders):
    images, masks = training_folders
    augmentation = config.AugmentationSettings(use_data_augmentation=True)
    data = train.load_training_data(images, masks, _settings(), augmentation)
    assert data.patch_size == 64 and data.steps_per_epoch == 2
    assert (len(data.split.x_train), len(data.split.x_val)) == (3, 1)
    folder = tmp_path / "models" / "tiny"
    epochs = []
    result = train.train(
        train.new_model(data, folder, overwrite=False), data, 1, True, lambda e, logs: epochs.append(logs)
    )
    assert len(epochs) == 1 and "loss" in epochs[0]
    evaluation = train.write_training_evaluation(folder, result.history)
    assert evaluation.read_text().splitlines()[0] == "loss,val_loss,learning rate"
    reloaded = model_module.load_model_folder(folder)
    assert reloaded.config.n_rays == 16
    assert (folder / "thresholds.json").is_file()
    with pytest.raises(FileExistsError, match="already exists"):
        train.new_model(data, folder, overwrite=False)


def test_quality_control_writes_predictions_and_the_table(tmp_path, training_folders, pretrained_folder):
    images, masks = training_folders
    star_dist = model_module.load_model_folder(pretrained_folder)
    result = qc.run_quality_control(star_dist, tmp_path / "model", images, masks)
    assert result.table.name == "Quality_Control for model.csv"
    assert sorted(path.name for path in result.predictions.iterdir()) == [f"cell_{i}.tif" for i in range(4)]
    assert all(row["Prediction v. GT Intersection over Union"] > 0.5 for row in result.rows)


def test_bioimage_model_zoo_export(tmp_path, pretrained_folder):
    star_dist = model_module.load_model_folder(pretrained_folder)
    metadata = bioimage.ModelMetadata("Versatile test", bioimage.parse_list("[Ada, Grace]"), ("Lab A",), "A test.")
    image = tifffile.imread(_nuclei(tmp_path))
    output = bioimage.export_model(star_dist, tmp_path / "export" / "model.zip", image, metadata)
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        spec = archive.read("rdf.yaml").decode()
    assert any(name.endswith(".h5") or "saved_model" in name or name.endswith(".zip") for name in names)
    assert "Versatile test" in spec and "Ada" in spec and "Lab A" in spec


def _nuclei(tmp_path):
    from stardist.data import test_image_nuclei_2d

    path = tmp_path / "nuclei.tif"
    tifffile.imwrite(path, test_image_nuclei_2d()[:256, :256])
    return path


def test_metadata_parsing_and_naming():
    assert bioimage.parse_list("[Author 1, Author 2, Author 3]") == ("Author 1", "Author 2", "Author 3")
    assert bioimage.default_output(__import__("pathlib").Path("/m/my_model")).name == "my_modelbioimageio.zip"
    with pytest.raises(ValueError, match="name"):
        bioimage.ModelMetadata(" ", ("A",))
    overrides = bioimage.spec_overrides(bioimage.ModelMetadata("m", ("A", "B"), ("X",)))
    assert overrides["authors"] == [{"name": "A", "affiliation": "X"}, {"name": "B"}]
    assert np.isclose(len(overrides["cite"]), 3)
