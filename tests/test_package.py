"""The package imports without TensorFlow and selects Keras 2 before TensorFlow is imported."""

from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

import stardist2d
from stardist2d import keras_backend

ROOT = Path(__file__).resolve().parents[1]


def _run(code: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run *code* in a fresh interpreter with src/ on the path (so module state starts clean)."""
    full_env = {key: value for key, value in os.environ.items() if key != "TF_USE_LEGACY_KERAS"}
    full_env.update({"PYTHONPATH": str(ROOT / "src"), **(env or {})})
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=full_env, check=False)


def test_import_is_light_and_selects_legacy_keras():
    result = _run(
        "import os, sys, stardist2d; "
        "print(os.environ['TF_USE_LEGACY_KERAS'], 'tensorflow' in sys.modules, 'stardist' in sys.modules)"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["1", "False", "False"]


def test_an_explicit_choice_is_respected():
    result = _run("import os, stardist2d; print(os.environ['TF_USE_LEGACY_KERAS'])", env={"TF_USE_LEGACY_KERAS": "0"})
    assert result.stdout.strip() == "0"


def _fake_tensorflow(layer_module):
    """A stand-in for TensorFlow whose tf.keras.layers.Layer comes from *layer_module* (as the real check reads)."""
    layer = type("Layer", (), {"__module__": layer_module})
    return types.SimpleNamespace(keras=types.SimpleNamespace(layers=types.SimpleNamespace(Layer=layer)))


def test_tensorflow_imported_first_with_keras_3_is_refused(monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorflow", _fake_tensorflow("tf_keras.src.engine.base_layer"))
    monkeypatch.setenv("TF_USE_LEGACY_KERAS", "0")
    with pytest.raises(keras_backend.KerasBackendError, match="Import stardist2d first"):
        keras_backend.check_keras_backend()
    monkeypatch.setenv("TF_USE_LEGACY_KERAS", "1")
    keras_backend.check_keras_backend()
    # The variable set too late: TensorFlow already loaded Keras 3 (audit finding 10).
    monkeypatch.setitem(sys.modules, "tensorflow", _fake_tensorflow("keras.src.layers.layer"))
    with pytest.raises(keras_backend.KerasBackendError, match="already uses Keras 3"):
        keras_backend.check_keras_backend()


def test_version_is_pep440():
    from packaging.version import Version

    assert str(Version(stardist2d.__version__)) == stardist2d.__version__
