"""Choose Keras 2 (``tf-keras``) before anything imports TensorFlow.

StarDist 0.9.2 and CSBDeep 0.8.2 run on Keras 2 through ``tf-keras`` when
``TF_USE_LEGACY_KERAS=1``, as ZeroCostDL4Mic and DL4MicEverywhere do. Under Keras 3,
CSBDeep's ``export_SavedModel`` raises ``NotImplementedError``, which breaks the
BioImage Model Zoo export. CSBDeep reads the variable when it is imported, and no
LabConstrictor host (Fiji, napari, QuPath) sets it for the worker, so the package sets
it on import.
"""

from __future__ import annotations

import os
import sys

LEGACY_KERAS_VARIABLE = "TF_USE_LEGACY_KERAS"
"""The environment variable TensorFlow and CSBDeep read to choose Keras 2 (``tf-keras``)."""


class KerasBackendError(RuntimeError):
    """TensorFlow was imported with Keras 3 before this package could choose Keras 2."""


def use_legacy_keras() -> None:
    """Select Keras 2 for TensorFlow and CSBDeep, unless the person chose explicitly.

    An explicit ``TF_USE_LEGACY_KERAS=0`` is respected (``setdefault``): someone who sets
    it knows the BioImage Model Zoo export will not work.
    """
    os.environ.setdefault(LEGACY_KERAS_VARIABLE, "1")


def check_keras_backend() -> None:
    """Refuse to go on when TensorFlow was already imported before Keras 2 was selected.

    Raises
    ------
    KerasBackendError
        If ``tensorflow`` is imported but ``TF_USE_LEGACY_KERAS`` is not ``1``, or TensorFlow already loaded
        Keras 3 (the variable was set too late).
    """
    if "tensorflow" not in sys.modules:
        return
    if os.environ.get(LEGACY_KERAS_VARIABLE) != "1":
        raise KerasBackendError(
            "TensorFlow was imported before stardist2d could select Keras 2. "
            "Import stardist2d first, or set TF_USE_LEGACY_KERAS=1 before starting Python."
        )
    # The variable only counts if it was set before TensorFlow loaded its Keras (it does so on the first use of
    # tf.keras): ask TensorFlow which Keras it actually uses, rather than trusting the variable.
    import tensorflow as tf

    # tf.keras and its submodules are lazy placeholders; a real class shows which Keras TensorFlow loaded
    # (Keras 2: tf_keras.src...; Keras 3: keras.src...).
    module = tf.keras.layers.Layer.__module__
    if not module.startswith("tf_keras"):
        raise KerasBackendError(
            f"TensorFlow already uses Keras 3 ({module}) in this Python session, so StarDist cannot use Keras 2. "
            "Restart the kernel (or the program) and import stardist2d before anything uses TensorFlow."
        )
