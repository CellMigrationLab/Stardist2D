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
- `stardist2d.data`, `stardist2d.prepare` (cells 17, 28), `stardist2d.model` and `stardist2d.predict` (cells 41, 43):
  prediction runs tile by tile with progress and cancel hooks. Folder prediction writes outputs identical to
  1.20.8's (labels, ImageJ ROI zips, centre and count CSVs; tested against the original cell).
- `stardist2d.batch` (stacks, big images), `stardist2d.pretrained` (section 3.4), `stardist2d.train` (sections 4.1,
  4.2, with a per-epoch callback), `stardist2d.qc` (section 5) and `stardist2d.bioimage` (section 5.3).
- Parity tests that run the code of the vendored 1.20.8 notebook (`tests/reference/`) and compare.

### Differences from ZeroCostDL4Mic 1.20.8
- **Patch size:** a patch size that is not a multiple of 16 is rounded down to the nearest multiple of 16
  (1.20.8 removed a further 16 pixels: 500 gave 480, now 496). The message now says "divisible by 16" (it said 8).
- **Training pairs:** every image needs a mask of the same name and the reverse; 1.20.8 ignored unmatched files.
- **Existing model folder:** deleted only after confirmation (1.20.8 deleted it without asking).
- **Fiji ROIs (section 6.1, single images):** the "Region_of_interests" box is honoured (1.20.8 always wrote the
  ROI zip); the box now defaults to on, so the default outputs are unchanged.
- **Stacks (section 6.1):** frames are normalised to the stack's 1-99.8 percentiles correctly (1.20.8 divided by
  (low - high), giving negative intensities).
- **Big images (section 6.2):** the intensity range comes from the image's percentiles (1.20.8 used 0-255 for every
  image, wrong for 16-bit images).
- **BioImage Model Zoo export:** the name, authors, affiliations, description, licence and references asked in
  section 5.3 are written into the model's description (1.20.8 asked for them but never passed them on).
  `importlib-metadata` is now a requirement: StarDist's exporter needs it and 1.20.8's requirements lacked it
  (the export only worked where it happened to be installed, such as Colab).
- **Training image size:** the patch size and default steps use the first image in name order (1.20.8 used a
  randomly chosen image).
- **Stack labels** are written as integers (1.20.8 converted them to 32-bit float), and the stack ROI zip comes from
  the same prediction as the labels (1.20.8 predicted every frame twice, the second time without normalisation).
- **QC table:** images are listed in sorted order (1.20.8 used the folder's listing order). Values are unchanged.

### Removed
- The LabConstrictor template's own tests and tool settings (`tests/test_template_sync_*`, `ruff.toml`,
  `mypy.ini`, `.github/workflows/tests.yml`): they test the template and run only in the template repository.
