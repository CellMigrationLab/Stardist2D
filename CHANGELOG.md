# Changelog

StarDist2D is a LabConstrictor port of the ZeroCostDL4Mic StarDist 2D notebook, version 1.20.8.
Every intentional difference from 1.20.8 is listed here under **Differences from ZeroCostDL4Mic 1.20.8**.

## Unreleased

### Added
- MorphoBricks code standard: the shared `tools/code_rules.py`, `tools/check.py`, an empty baseline and
  the `StarDist2D tests` workflow.
- `stardist2d` package skeleton; importing it selects Keras 2 (`TF_USE_LEGACY_KERAS=1`) without importing
  TensorFlow.

### Removed
- The LabConstrictor template's own tests and tool settings (`tests/test_template_sync_*`, `ruff.toml`,
  `mypy.ini`, `.github/workflows/tests.yml`): they test the template and run only in the template repository.
