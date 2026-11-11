"""StarDist 2D (ZeroCostDL4Mic port) for LabConstrictor: training, quality control and prediction.

Importing the package is light (no TensorFlow): the LabConstrictor installer checks
``import stardist2d`` at the end of every installation. It selects Keras 2 for TensorFlow
(:mod:`stardist2d.keras_backend`) before any module imports it.
"""

from __future__ import annotations

from stardist2d.keras_backend import use_legacy_keras

use_legacy_keras()

__version__ = "1.0.0.dev0"
"""Package version; moves with the notebook's ``current_version`` and the release tag."""

__all__ = ["__version__"]
