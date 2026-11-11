"""The command line runs the same steps as the notebook."""

import pytest

from stardist2d import cli, provenance


def test_an_expected_problem_gives_exit_code_1_and_one_line(tmp_path, capsys):
    assert cli.main(["qc", str(tmp_path / "no_model"), str(tmp_path), str(tmp_path)]) == 1
    assert capsys.readouterr().err.startswith("stardist2d: ")


@pytest.mark.slow
def test_train_then_predict_from_the_command_line(tmp_path, training_folders, pretrained_folder):
    images, masks = training_folders
    folder = tmp_path / "models" / "tiny"
    assert (
        cli.main(
            [
                "train",
                str(images),
                str(masks),
                str(folder),
                "--epochs",
                "1",
                "--advanced",
                "--steps",
                "2",
                "--patch-size",
                "64",
                "--augment",
                "1",
            ]
        )
        == 0
    )
    assert (folder / "Quality Control" / "training_evaluation.csv").is_file()
    assert provenance.read_run_file(folder)["step"] == "train"
    results = tmp_path / "results"
    assert cli.main(["predict", str(pretrained_folder), str(images), str(results), "--tiles"]) == 0
    assert (results / "Nuclei_count.csv").is_file() and (results / "cell_0.zip").is_file()
    assert provenance.read_run_file(results)["step"] == "predict images"
