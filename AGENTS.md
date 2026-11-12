# AGENTS.md — StarDist2D

Rules for anyone (person or AI agent) changing this repository. Read this first; prefer evidence in the
repository over assumptions. The full plan and its decisions are in [docs/plan.md](docs/plan.md).

## What this repository is

- A LabConstrictor app built from the LabConstrictor template: a port of the ZeroCostDL4Mic **StarDist 2D**
  notebook (version 1.20.8). It runs locally (LabConstrictor installer, JupyterLab) and on Google Colab.
- The science lives in the package `src/stardist2d`. The notebook and `src/stardist2d_lc_tools` (the
  LabConstrictor-Tools bridge to Fiji, napari and QuPath) only call it.
- Template files are synced from LabConstrictor and are not edited by hand: `app/`, `.tools/`,
  `.template_sync/`, the generated `requirements*.txt` / `*.lock.txt`, `construct.yaml`, and the template
  workflows. Our own CI is `.github/workflows/stardist2d_tests.yml`.

## Fixed technical choices (do not change without the owner's approval)

- Python 3.12; TensorFlow 2.20 + tf-keras 2.20 (Keras 2, `TF_USE_LEGACY_KERAS=1`, set by
  `stardist2d.keras_backend` on import); stardist 0.9.2; csbdeep 0.8.2. The same stack as ZeroCostDL4Mic and
  DL4MicEverywhere, so models move between the three. Keras 3 breaks the BioImage Model Zoo export.
- Platforms: Windows x86_64 (CPU), macOS Apple Silicon (CPU), Linux x86_64 (CPU or NVIDIA GPU). No Intel Mac.
- `import stardist2d` stays light (no TensorFlow): the installer checks it at the end of every installation.
- Bugs of 1.20.8 are fixed, and every difference from 1.20.8 is listed in `CHANGELOG.md` with a test.
- Output file names and CSV columns of the original notebook are kept (users' macros depend on them).

## MorphoBricks code standard (enforced)

`python tools/check.py` runs pytest, the shared code rules (`tools/code_rules.py`, byte-identical in every
MorphoBricks repository, SHA-tested) and Ruff, against a shrink-only baseline that starts **empty**:

- ≤600 lines per file, ≤60 executable lines per function, ≤8 parameters, complexity ≤15, no
  `contextlib.suppress`, no duplicated function bodies, no import cycles, private functions over 10 lines
  have a docstring.
- Every module of `src/stardist2d` declares one layer in `pyproject.toml`: `pure` (no I/O, no UI),
  `io`, `ui` (no computing) or `orchestration`.
- Do not define names owned by other bricks (`RESERVED_NAMES` in `tools/code_rules.py`), e.g.
  `source_fingerprint`, `check_labels`, `read_mask`, `percentile_range`, `min_max`.
- `from __future__ import annotations`; full type annotations; NumPy docstrings with an imperative first
  line; comments explain why, not what.
- Raise descriptive exceptions that say what to do; refuse rather than guess; never fail silently.
- No processing logic in notebook cells; no `!`/`%` magics; Colab-only code inside
  `if "google.colab" in sys.modules:`.
- Every output folder gets a provenance file; writes are atomic; never write into an input folder.
- Tests use small synthetic data in `tmp_path`. Tests that need TensorFlow are marked `slow`; never delete
  or weaken a test.

## Working rules

- Make the smallest coherent change; list unrelated clean-ups as follow-ups.
- Never claim a check passed unless it was run; report what was run and its limits.
- Ask before: public API or output-format changes, new required dependencies, changes that alter
  scientific results, destructive operations.
- `CHANGELOG.md` gets an entry for every user-visible change.

## Releasing a new version

1. On a branch: set the new version in `setup.py`, `src/stardist2d/__init__.py` (`__version__`), `construct.yaml`
   (`version:` and the conclusion text) and `tools/build_notebook.py` (`VERSION`), run `python tools/build_notebook.py`,
   and give `CHANGELOG.md` a dated section. `tests/test_versions.py` fails while any of them differ. Merge it.
2. Wait until no workflow is running, then publish a GitHub release whose tag is the bare version (e.g. `1.0.1`, no
   `v`): the template's download page links to `releases/download/<version>/`, and the Colab install cell installs the
   package from that tag. The release workflow builds the installers.
