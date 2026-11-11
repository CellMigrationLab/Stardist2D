"""Files written by notebook sections 5.2 and 6.1 (ZeroCostDL4Mic 1.20.8, cells 36 and 41).

File names, CSV headers and number formats are those of 1.20.8, so macros and scripts written for
its outputs keep working. Every write goes to a temporary file first and is then moved into place,
so an interrupted run never leaves a half-written file behind.
"""

from __future__ import annotations

import csv
import os
import tempfile
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from stardist2d.metrics import QC_COLUMNS


def _atomic_write(path: Path, write: Callable[[Path], None]) -> Path:
    """Call ``write(temporary_path)`` and move the result to *path* only when it succeeded."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=path.suffix, dir=path.parent)
    os.close(handle)
    try:
        write(Path(temporary))
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return path


def _write_csv(path: Path, rows: Iterable[Sequence[Any]], line_end: str = "\r\n") -> Path:
    """Write *rows* as CSV with *line_end* after each row.

    1.20.8 writes the QC table with the ``csv`` module (CR LF) and the prediction tables with pandas
    (LF); each writer below keeps the line ending of the file it replaces.
    """

    def write(temporary: Path) -> None:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator=line_end)
            writer.writerows(rows)

    return _atomic_write(path, write)


def write_labels(path: Path, labels: np.ndarray) -> Path:
    """Write a label image as TIFF, keeping its integer type (1.20.8 writes ``model.predict_instances`` labels)."""
    import tifffile

    return _atomic_write(path, lambda temporary: tifffile.imwrite(temporary, labels))


def write_imagej_rois(path: Path, polygons: np.ndarray | Sequence[np.ndarray]) -> Path:
    """Write StarDist polygons (``details['coord']``, shape ``(n, 2, n_rays)``) as an ImageJ ROI zip.

    *path* ends in ``.zip``. The ROIs are those of 1.20.8 (``stardist.export_imagej_rois``). A list of
    polygon arrays (one per frame of a stack) gives each frame's ROIs that frame as position.
    """
    from stardist import export_imagej_rois

    if path.suffix != ".zip":
        raise ValueError(f"an ImageJ ROI file must end in .zip, got {path.name}")

    def write(temporary: Path) -> None:
        # export_imagej_rois adds ".zip" to the name it is given.
        export_imagej_rois(str(temporary.with_suffix("")), polygons)

    return _atomic_write(path, write)


def write_centres(path: Path, points: np.ndarray) -> Path:
    """Write object centres (``details['points']``, rows of ``(y, x)``) as ``<name>_Nuclei_centre.csv``."""
    return _write_csv(path, [("Y", "X"), *(tuple(point) for point in np.asarray(points).tolist())], "\n")


def write_counts(path: Path, counts: Iterable[tuple[str, int]]) -> Path:
    """Write ``(name, number of objects)`` rows without a header, as 1.20.8's ``Nuclei_count.csv``."""
    return _write_csv(path, counts, "\n")


def write_qc_table(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    """Write quality-control rows (:func:`stardist2d.metrics.qc_row`) with 1.20.8's header."""
    return _write_csv(path, [QC_COLUMNS, *([row[column] for column in QC_COLUMNS] for row in rows)])


def print_table(rows: Sequence[dict[str, Any]], stream: TextIO) -> None:
    """Print rows as the ``psql`` table 1.20.8 shows below the QC cell."""
    import pandas as pd
    from tabulate import tabulate

    print(tabulate(pd.DataFrame(rows), headers="keys", tablefmt="psql"), file=stream)
