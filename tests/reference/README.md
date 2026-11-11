# Reference: ZeroCostDL4Mic StarDist 2D 1.20.8

`StarDist_2D_ZeroCostDL4Mic_1.20.8.ipynb` is a byte-exact copy of
`Colab_notebooks/StarDist_2D_ZeroCostDL4Mic.ipynb` from
[HenriquesLab/ZeroCostDL4Mic](https://github.com/HenriquesLab/ZeroCostDL4Mic) at commit
`bd7bc0a0f493bc9c7c3f91d260266b679dbdfa5a` (2026-09-28), the notebook this app ports (MIT licence).

It is never edited. The parity tests (`tests/parity/`) run code taken from its cells, so they compare
this package with what 1.20.8 really does, and the notebook tests check that every original form
parameter and every markdown cell is carried over. `tests/test_reference.py` checks the SHA-256 below.

SHA-256: see `tests/test_reference.py`.

Ruff skips this folder (`extend-exclude`): it is upstream code, not ours.
