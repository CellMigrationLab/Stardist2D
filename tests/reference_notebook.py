"""Run code taken from the cells of the vendored ZeroCostDL4Mic 1.20.8 notebook (tests/reference).

The parity tests compare the package with what 1.20.8 really does: they execute the original
statements and functions of a cell, never a retyped copy.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable
from functools import cache
from pathlib import Path

NOTEBOOK = Path(__file__).parent / "reference" / "StarDist_2D_ZeroCostDL4Mic_1.20.8.ipynb"
_MAGIC = re.compile(r"^(\s*)[!%].*$", re.MULTILINE)


@cache
def cells() -> list[dict]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def source(index: int) -> str:
    """Return the source of cell *index*, with IPython magics (``!``/``%`` lines) replaced by ``pass``."""
    return _MAGIC.sub(r"\1pass", "".join(cells()[index]["source"]))


def statements(index: int, keep: Callable[[str], bool]) -> str:
    """Return the top-level statements of cell *index* whose source satisfies *keep*, as code."""
    text = source(index)
    tree = ast.parse(text)
    segments = [ast.get_source_segment(text, node) or "" for node in tree.body]
    return "\n".join(segment for segment in segments if keep(segment))


def run(index: int, keep: Callable[[str], bool], namespace: dict) -> dict:
    """Execute the kept statements of cell *index* in *namespace* and return it."""
    exec(compile(statements(index, keep), f"<cell {index}>", "exec"), namespace)
    return namespace


def functions(index: int, names: tuple[str, ...], namespace: dict) -> dict:
    """Define the functions *names* of cell *index* in *namespace* and return it."""
    wanted = re.compile(r"^def ({})\(".format("|".join(map(re.escape, names))))
    return run(index, lambda code: bool(wanted.match(code)), namespace)
