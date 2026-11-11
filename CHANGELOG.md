# Changelog

StarDist2D is a LabConstrictor port of the ZeroCostDL4Mic StarDist 2D notebook, version 1.20.8.
Every intentional difference from 1.20.8 is listed here under **Differences from ZeroCostDL4Mic 1.20.8**.

## Unreleased

### Added
- MorphoBricks code standard: the shared `tools/code_rules.py`, `tools/check.py`, an empty baseline and
  the `StarDist2D tests` workflow.
- `stardist2d` package skeleton; importing it selects Keras 2 (`TF_USE_LEGACY_KERAS=1`) without importing
  TensorFlow.

- `stardist2d.config` (training settings and the rules of sections 3.2–4.1), `stardist2d.augment` (section 3.3),
  `stardist2d.metrics` (section 5.2 metrics) and `stardist2d.export` (QC table, label images, ImageJ ROIs,
  centre and count CSVs, written atomically).
- Parity tests that run the code of the vendored 1.20.8 notebook (`tests/reference/`) and compare.

### Differences from ZeroCostDL4Mic 1.20.8
- **Patch size:** a patch size that is not a multiple of 16 is rounded down to the nearest multiple of 16
  (1.20.8 removed a further 16 pixels: 500 gave 480, now 496). The message now says "divisible by 16" (it said 8).
- **QC table:** images are listed in sorted order (1.20.8 used the folder's listing order). Values are unchanged.

### Removed
- The LabConstrictor template's own tests and tool settings (`tests/test_template_sync_*`, `ruff.toml`,
  `mypy.ini`, `.github/workflows/tests.yml`): they test the template and run only in the template repository.
