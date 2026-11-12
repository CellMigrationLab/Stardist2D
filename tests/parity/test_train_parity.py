"""Training files and learning-rate rules equal those of cells 25 and 30 of 1.20.8."""

import csv

import pandas as pd
import pytest

from reference_notebook import run, source
from stardist2d import pretrained, train


class _History:
    def __init__(self, history):
        self.history = history


@pytest.mark.parametrize("rate_key", ["learning_rate", "lr", None])
def test_training_evaluation_csv_matches_1_20_8(tmp_path, rate_key):
    history = {"loss": [1.5, 1.2, 1.1], "val_loss": [1.6, 1.0, 1.05]}
    if rate_key:
        history[rate_key] = [0.0003, 0.0003, 0.00015]
    namespace = {"lossDataCSVpath": str(tmp_path / "reference.csv"), "history": _History(history), "csv": csv}
    run(30, lambda code: code.startswith("with open(lossDataCSVpath"), namespace)
    ours = train.write_training_evaluation(tmp_path / "model", history)
    assert ours.read_bytes() == (tmp_path / "reference.csv").read_bytes()


@pytest.mark.parametrize("weights", ["last", "best"])
def test_learning_rate_of_a_pretrained_model_matches_1_20_8(tmp_path, weights):
    qc = tmp_path / "model" / "Quality Control"
    qc.mkdir(parents=True)
    pd.DataFrame({"loss": [3, 2, 1], "val_loss": [2.0, 0.5, 0.9], "learning rate": [0.001, 0.0005, 0.0001]}).to_csv(
        qc / "training_evaluation.csv", index=False
    )
    namespace = {
        "pretrained_model_path": str(tmp_path / "model"),
        "Weights_choice": weights,
        "os": __import__("os"),
        "pd": pd,
        "print": lambda *args, **kwargs: None,
        "initial_learning_rate": 0.0003,
        "bcolors": type("C", (), {"WARNING": "", "ENDC": ""}),
    }
    block = "if os.path.exists(os.path.join(pretrained_model_path, 'Quality Control', 'training_evaluation.csv')):"
    text = source(25)
    start = text.rindex("\n", 0, text.index(block)) + 1  # from the start of the line (4 spaces of indent)
    block_lines = text[start:].split("#Compatibility")[0].splitlines()
    code = "\n".join(line[4:] for line in block_lines)
    exec(code, namespace)
    expected = namespace["lastLearningRate"] if weights == "last" else namespace["bestLearningRate"]
    assert pretrained.learning_rate(tmp_path / "model", weights, 0.0003)[0] == expected


def test_a_model_without_training_table_uses_the_default_rate(tmp_path):
    assert pretrained.learning_rate(tmp_path, "best", 0.0003)[0] == 0.0003


def test_weights_file_names_follow_1_20_8(tmp_path):
    for name in ("weights_best.h5", "weights_last.h5", "weights_bioimageio.h5"):
        (tmp_path / name).write_bytes(b"")
    assert pretrained.weights_file("Model_from_file", tmp_path, "last").name == "weights_last.h5"
    assert pretrained.weights_file("2D_versatile_fluo_from_Stardist_Fiji", tmp_path, "last").name == "weights_best.h5"
    assert pretrained.weights_file("BioImage Model Zoo", tmp_path, "best").name == "weights_bioimageio.h5"
    with pytest.raises(FileNotFoundError, match="do not exist"):
        pretrained.weights_file("Model_from_file", tmp_path / "missing", "best")
