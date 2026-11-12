# Changelog

StarDist2D is a LabConstrictor port of the ZeroCostDL4Mic StarDist 2D notebook, version 1.20.8.
Every intentional difference from 1.20.8 is listed here under **Differences from ZeroCostDL4Mic 1.20.8**.

## 1.0.1 (2026-11-12)

### Fixed
- **Installation failed on every system.** The installer's last step ran `import PYTHON_PROJ_NAME`: the repository
  initialisation had left the template's placeholder for the package name in `post_install.sh` and `post_install.bat`.
  They now check `stardist2d` and register `stardist2d_lc_tools`; the JupyterLab launcher's error messages name
  StarDist2D. `tests/test_installer_scripts.py` fails if a placeholder is left in an installed script.
- **Download links:** release tags are now the bare version (`1.0.1`, not `v1.0.1`), as the template's download page
  expects. The Colab install cell installs from that tag.

## 1.0.0 (2026-11-12)

First release: the ZeroCostDL4Mic StarDist 2D notebook 1.20.8 as a LabConstrictor app, with its code in the
`stardist2d` package and tools for Fiji, napari and QuPath.

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
- `stardist2d.report` (training and QC PDFs from explicit inputs), `stardist2d.plotting` (the notebook's figures),
  `stardist2d.environment` (versions, GPU, CUDA without IPython magics), `stardist2d.provenance`
  (`stardist2d_run.json` beside every output) and a command line (`python -m stardist2d train|qc|predict`).
- **The notebook** `notebooks/StarDist_2D/StarDist_2D_ZeroCostDL4Mic.ipynb`, built by `tools/build_notebook.py` from
  the 1.20.8 notebook: every markdown cell kept word for word (plus "JupyterLab users" notes on Colab-only steps),
  every code cell one call of `stardist2d.notebook`, which shows a form with 1.20.8's parameter names and defaults.
  It runs locally (LabConstrictor, JupyterLab, ipywidgets 8) and on Colab (ipywidgets 7.7).
- A small example dataset shipped with the installer (`example_data/`, from Zenodo 7949940) so section 3.1 works
  offline; installer images from the StarDist logo.
- **Fiji, napari and QuPath tools** (`src/stardist2d_lc_tools`, LabConstrictor-Tools bridge): Segment nuclei
  (fluorescence, with channel choice and region runs), Segment nuclei (H&E / RGB), Segment a folder, Quality control,
  Check this installation. Outputs: 16-bit labels, StarDist's own outlines (ROI Manager / Shapes / annotations) with
  label and probability, a per-object table. Prediction runs tile by tile with progress and Cancel; loaded models are
  cached. `labconstrictor-app.yaml` lists the app in the notebook, napari, Fiji and QuPath.
- `lc_tests/cases.json`: the tools' test cases (`labconstrictor-tools test`), also run by pytest with a Cancel test.
- Section 3.1 can also download nine StarDist datasets of the Cell Migration Lab from Zenodo (`stardist2d.datasets`:
  HUVEC nuclei, brightfield neutrophils / monocytes / cancer cells 20x and 10x, fluorescent cells, AsPC1, MiaPaCa2,
  tumour cell nuclei; CC BY 4.0), each with the model trained on it, and prints the paths to use in sections 3.2–6.
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
- **Reports:** the Augmentor reference is no longer cited (the augmentation never used Augmentor); the GPU name
  comes from `nvidia-smi --query-gpu=name` (1.20.8 only recognised "Tesla" GPUs); a missing loss-curve figure no
  longer stops the QC report.
- **Forms instead of Colab fields:** Colab `#@param` fields do not exist in JupyterLab; each cell shows the same
  fields as widgets and a Run button. Section 6.2 needs equal block_size_Y and block_size_X.
- **Colour (multi-channel) images:** normalised with all channels together for training too, as for quality control
  and prediction (1.20.8 trained with each channel normalised on its own, then predicted with them together).
  Single-channel images are unchanged.
- **QC table:** images are listed in sorted order (1.20.8 used the folder's listing order). Values are unchanged.

### Removed
- The LabConstrictor template's own tests and tool settings (`tests/test_template_sync_*`, `ruff.toml`,
  `mypy.ini`, `.github/workflows/tests.yml`): they test the template and run only in the template repository.
