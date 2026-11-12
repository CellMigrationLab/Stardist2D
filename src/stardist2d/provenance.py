"""A ``stardist2d_run.json`` file beside every output, so each result can be traced to its inputs and settings.

New in this port (1.20.8 kept only a ``requirements.txt``). The file is versioned and written atomically.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import stardist2d
from stardist2d.environment import EnvironmentInfo

RUN_FILE = "stardist2d_run.json"
RUN_FILE_VERSION = 1


def file_digest(path: Path) -> str:
    """Return the SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain(value: object) -> object:
    """Return *value* as JSON-ready data (dataclasses, paths and tuples included)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _plain(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    return value


def write_run_file(
    folder: Path, step: str, settings: dict[str, Any], inputs: list[Path], environment: EnvironmentInfo
) -> Path:
    """Write ``<folder>/stardist2d_run.json``: what ran, with which settings, on which files (with digests)."""
    record = {
        "run_file_version": RUN_FILE_VERSION,
        "step": step,
        "created": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "stardist2d_version": stardist2d.__version__,
        "port_of": "ZeroCostDL4Mic StarDist 2D 1.20.8",
        "settings": _plain(settings),
        "inputs": [{"file": path.name, "sha256": file_digest(path)} for path in sorted(inputs)],
        "environment": _plain(environment),
    }
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / RUN_FILE
    handle, temporary = tempfile.mkstemp(prefix=f".{RUN_FILE}.", dir=folder)
    with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, target)
    return target


def read_run_file(folder: Path) -> dict[str, Any]:
    """Return the run file of *folder*.

    Raises
    ------
    ValueError
        If the file was written by a newer version of the format.
    """
    record = json.loads((folder / RUN_FILE).read_text(encoding="utf-8"))
    if record.get("run_file_version") != RUN_FILE_VERSION:
        raise ValueError(
            f"{folder / RUN_FILE} has format version {record.get('run_file_version')}; this version reads 1"
        )
    return record
