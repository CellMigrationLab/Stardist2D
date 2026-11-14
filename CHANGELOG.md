# Changelog

StarDist2D is a LabConstrictor port of the ZeroCostDL4Mic StarDist 2D notebook, version 1.20.8.
Every intentional difference from 1.20.8 is listed here under **Differences from ZeroCostDL4Mic 1.20.8**.

## 1.2.1 (2026-11-14)

### Changed
- **The app says which plugin versions it needs** (`host_requirements` in `labconstrictor-app.yaml`, LabConstrictor-Tools
  #155): LabConstrictor-napari 0.1.0-alpha.5, LabConstrictor-Fiji 0.1.0-alpha.7 (progress of long trainings in the Log
  window) and LabConstrictor-QuPath 0.1.0-alpha.7 (Replace keeps your own annotations). The installer registers it; a
  plugin that checks it and is older tells the person to update, and the app page on the LabConstrictor website lists
  the versions.
- Template 0.1.28: the installer installs LabConstrictor-Tools v0.3.0-alpha.2 (it registers the requirements above).

## 1.2.0 (2026-11-14)

### Added
- **Ten more pretrained models**, in every model list:
  - the Fiji, napari and QuPath tools (*Model*; *Start from* in *Train a model*);
  - the notebook (section 3.4, and a new *Pretrained_model* choice in sections 6.1 and 6.2);
  - the command line.

  Nine are the Cell Migration Lab's StarDist models, taken unchanged from the model folders of its Zenodo datasets
  (CC BY 4.0): HUVEC nuclei, fluorescent cells, AsPC1 (Lifeact), MiaPaCa2 (CD44), tumour cell nuclei (DAPI),
  neutrophils, monocytes, and cancer cells at 20x and 10x (brightfield). The tenth is the BioImage Model Zoo's CHO
  mitotic rounding model (brightfield, MIT). They are rehosted in `models/` (about 5.3 MB each, instead of
  100–580 MB per dataset) and registered in StarDist's own registry: downloaded once, checked against their SHA-256,
  cached like `2D_versatile_fluo`. The summaries and the notebook say what to cite. See `models/README.md`.

### Fixed
- **A pretrained model that cannot be downloaded or fails its checksum is refused with what to do.** CSBDeep's
  `from_pretrained` printed a table and returned nothing, which surfaced later as "'NoneType' object has no
  attribute …". This affected StarDist's own models too, for example offline. The tools report it as
  `download_failed`.

## 1.1.2 (unreleased)

Fixes from the first human tests (Windows, notebook and Fiji).

### Fixed
- **Section 6.2, stacks and folders of big images:** the second frame of a stack (and the second image of a folder)
  failed with "array already exists": every frame was written to the same temporary zarr array. Each prediction now
  gets a fresh one. An image smaller than `block_size` is refused with what to do (StarDist raised a bare
  `AssertionError`).
- **Section 3.4, pretrained models:** the demo, versatile fluorescence and H&E models could not be downloaded; the
  1.20.8 addresses (cloud.mpi-cbg.de, the old mpicbg-csbd GitHub path) no longer serve them. They now come from the
  StarDist team's stardist-models releases through `StarDist2D.from_pretrained`, checked against StarDist's
  SHA-256. These archives hold only `weights_best.h5`, so "last" weights load the best weights for them. A failed
  download says where to get the model by hand.
- **Notebook:** the forms no longer print "Form(title=...)" under their cell (each step cell ends with `;`), and a
  running step shows "Running…" on its disabled button, since JupyterLab's busy indicator does not always turn on
  for a widget callback.
- **Train a model (tools):** a low validation F1 (below 0.5) comes with advice: more epochs, or start from
  `2D_versatile_fluo`.
- **Train a model (tools):** progress within each epoch ("epoch 2/5, step 7/16", at most once a second), not only
  after it: on a CPU an epoch can take minutes, and the hosts looked frozen in between.
- **Versions:** 1.1.1 was released from `main` without the version step, so only `construct.yaml` and `setup.py`
  said 1.1.1. Everything says 1.1.2 again.

## 1.1.0 (2026-11-12)

### Added
- **Train a model** (Fiji, napari, QuPath): the notebook's sections 3.2-4.2 as a tool.
  - **Inputs:** folders of training images and masks; a new model name; epochs; augmentation; a start from scratch,
    from a StarDist pretrained model or from your own model folder (section 3.4); the advanced parameters with the
    notebook's defaults.
  - **While it runs:** progress after every epoch.
  - **Outputs:** the loss table, the QC metrics on the validation images, and a summary naming the model folder,
    ready for *Custom model folder*.
  - **Model folder:** the same files as the notebook writes, plus `stardist2d_run.json`.
- **Cancel during training** stops within a batch (the hosts stop the worker 3 s after Cancel). The model folder keeps
  `config.json`, the best weights so far, `Quality Control/training_evaluation.csv` and `stardist2d_run.json`; the
  last two are rewritten after every epoch and say where training stopped. Thresholds are not optimised, so StarDist's
  defaults apply. 1.20.8 has no Cancel.
- **Quality control on a folder** (Fiji, napari, QuPath): the notebook's section 5 for a model folder. It writes the
  same `Quality_Control for <model>.csv` and predictions in the model's `Quality Control` folder.

### Changed
- `training_evaluation.csv` is written atomically (same content and bytes as before).
## 1.0.4 (2026-11-12)

### Changed
- **Citing StarDist:**
  - **Notebook introduction:** it cites the published StarDist paper, Schmidt *et al.*, *Cell Detection with
    Star-Convex Polygons*, MICCAI 2018, LNCS 11071, pp. 265-273,
    [doi:10.1007/978-3-030-00934-2_30](https://doi.org/10.1007/978-3-030-00934-2_30). 1.20.8 linked only the arXiv
    preprint, which is kept as the preprint link. These are the only edits to 1.20.8's markdown text
    (`TEXT_CHANGES` in `tools/build_notebook.py`, checked by `tests/test_notebook.py`).
- **LabConstrictor website:** the app's "How to cite" DOI is the StarDist paper. The description asks users to cite
  StarDist, ZeroCostDL4Mic and LabConstrictor.
- **README:** the Credits section gives the full references.

## 1.0.3 (2026-11-12, not released separately; included in 1.0.4)

Corrections from an independent audit of 1.0.2. Each fix has a test (`tests/test_audit_fixes.py`, `tests/test_batch.py`).

### Fixed
- **Original images could be overwritten** (notebook section 6 and the command line). A results folder that is, contains
  or lies inside the image folder (symbolic links resolved) is now refused before anything is written. 1.20.8 had the
  same problem; the Fiji, napari and QuPath tools already refused the same folder.
- **Resuming a stack** (section 6.1) checks the frames themselves. Frames `frame_00000.tif`, `frame_00001.tif`... are
  checked in order, and counting stops at the first missing or unreadable one. 1.20.8 counted every file in the
  folder, so a stray file or a half-written frame skipped frames that were never predicted. The resumed stack's
  `<stack>_Nuclei_number.csv` is complete: the counts of earlier frames come from their label images. 1.20.8 kept only
  the new frames.
- **Big-image stacks** (section 6.2) are skipped when complete and resumed when partial, with the same frame check, as
  1.20.8 did. 1.0.0–1.0.2 predicted every stack again.
- **The tools' model cache** reloads a model when its weights or thresholds change, not only its `config.json`.
- **Normalisation is checked:**
  - percentiles must satisfy 0 ≤ low < high ≤ 100 (reversed limits inverted the intensities);
  - images with NaN or infinite pixels are refused;
  - an image whose two percentiles are equal is refused, because its rare bright pixels would be amplified about
    10^20 times. A uniform image still gives no objects.
- **Thresholds and tiling are checked:** probability and overlap thresholds must lie between 0 and 1. A misspelt
  `n_tiles` (anything but `None`, `"auto"` or a tuple) is refused; it used to switch tiling off silently.
- **Undefined QC values are explained:** when ground truth and prediction are both empty, the IoU stays `NaN` in the
  QC table (as 1.20.8). The reason (`both_masks_empty`) is written beside it in
  `Quality_Control for <model>_undefined.csv` and named in the tool's summary.
- **Quality control (tools)** refuses more than 65,535 objects, instead of wrapping the 16-bit labels it returns.
- **Dataset downloads** (section 3.1) are unpacked into a staging folder. Every part (training and QC images and
  masks, the model) is checked, and only then is the folder marked complete. A folder left by an interrupted download
  is replaced, and a download shorter than announced is refused.
- **Pretrained-model downloads** (section 3.4) are streamed to a temporary file and checked for a `config.json` and
  weights before they replace anything.
- **Atomic writes:** `training_evaluation.csv`, the PDF reports and the saved figures are written to a temporary file
  first (same content).
- **Keras check:** TensorFlow already running Keras 3 in the same Python session (imported and used before
  `stardist2d`) is now refused with what to do. The check used to look only at the environment variable.
- **Versions:** the training report and the notebook's introduction say the current version (they said 1.0.0).
- **Section 5.3:** the `PixelSize` field says it is not used (1.20.8 asked for it and never passed it on).

### Documented differences from 1.20.8 not listed before
- **Section 6.1, resumed stacks:** no ROI zip is written, because the frames predicted earlier have no ROIs. The
  notebook and the tools say so.
- **Sections 6.1 and 6.2:** only `.tif` files of the input folders are read. 1.20.8 tried every file.
- **Section 6.2:** the final two-panel preview of the last label image is not shown.
- **Parity tests:** label TIFFs and ImageJ ROI zips are compared by their pixel values and ROI contents, not their
  file bytes. The CSV files are compared byte for byte.

### Known gaps (planned)
- A provenance file is written by command-line training and prediction only, not yet by the notebook, quality
  control or the batch workflows. Training provenance does not yet record the model's weights.
- Section 6.2 reads the whole image into memory before predicting it block by block. It is not out-of-core, so
  very large images need enough RAM.
- The parity tests run selected statements of the 1.20.8 cells, and QC parity starts from written predictions.
  End-to-end parity tests are planned.

## 1.0.2 (2026-11-12)

### Fixed
- **Windows installer script line endings:** `post_install.bat` has CR LF line endings again, as in the template (the
  repository initialisation had rewritten it with LF; `cmd.exe` can miss `GOTO`/`CALL` labels in an LF-only batch file).
  `tests/test_installer_scripts.py` checks it.
- **LabConstrictor website:** `labconstrictor-app.yaml` no longer sets `homepage` to the repository address (the site
  refused it as a second, plain link to the repository).

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
  prediction runs tile by tile with progress and cancel hooks. Folder prediction writes outputs matching
  1.20.8's (labels, ImageJ ROI zips, centre and count CSVs; tested against the original cell).
- `stardist2d.batch` (stacks, big images), `stardist2d.pretrained` (section 3.4), `stardist2d.train` (sections 4.1,
  4.2, with a per-epoch callback), `stardist2d.qc` (section 5) and `stardist2d.bioimage` (section 5.3).
- `stardist2d.report` (training and QC PDFs from explicit inputs), `stardist2d.plotting` (the notebook's figures),
  `stardist2d.environment` (versions, GPU, CUDA without IPython magics), `stardist2d.provenance`
  (`stardist2d_run.json`, written by command-line training and prediction) and a command line (`python -m stardist2d train|qc|predict`).
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
