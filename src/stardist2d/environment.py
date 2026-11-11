"""What the machine and the Python environment are (1.20.8, cells 8 and 11), without IPython magics."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from importlib import metadata

REPORTED_PACKAGES = ("tensorflow", "tf-keras", "stardist", "csbdeep", "numpy")


@dataclass(frozen=True)
class EnvironmentInfo:
    """Package versions and the GPU, for the reports and the provenance file."""

    versions: dict[str, str]
    gpu_name: str
    cuda_version: str


def package_version(name: str) -> str:
    """Return the installed version of *name*, or ``"not installed"``."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def _run(command: list[str]) -> str:
    """Return the standard output of *command*, or an empty string when it cannot run."""
    if shutil.which(command[0]) is None:
        return ""
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=30, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def gpu_name() -> str:
    """Return the NVIDIA GPU's name (``nvidia-smi``), or ``"No GPU found"``.

    1.20.8 searched the ``nvidia-smi`` table for the word "Tesla", which missed every other GPU.
    """
    names = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    return names.splitlines()[0] if names else "No GPU found"


def cuda_version() -> str:
    """Return the CUDA version TensorFlow was built with, or ``"No cuda found"`` (1.20.8 asked ``nvcc``)."""
    try:
        import tensorflow as tf
    except ImportError:
        return "No cuda found"
    return str(tf.sysconfig.get_build_info().get("cuda_version") or "No cuda found")


def collect() -> EnvironmentInfo:
    """Return the versions of the main packages, the GPU and CUDA."""
    return EnvironmentInfo({name: package_version(name) for name in REPORTED_PACKAGES}, gpu_name(), cuda_version())


def gpu_report() -> str:
    """Return section 2.1's answer: the ``nvidia-smi`` table, or a sentence saying there is no NVIDIA GPU."""
    table = _run(["nvidia-smi"])
    if table:
        return table
    return "You do not have a GPU (NVIDIA) available: training will run on the CPU and take longer."
