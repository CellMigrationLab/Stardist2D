# Plan v2 — StarDist 2D (ZeroCostDL4Mic) as a single LabConstrictor repository

Status: reviewed adversarially (independent re-check of every claim against code, PyPI and real dependency
resolution). Supersedes v1 (kept as `STARDIST2D_LABCONSTRICTOR_PLAN_v1.md`). Nothing is committed anywhere.
The draft manifest and tool declarations arrive in P3.

---

## 0. Decisions in this plan (and why)

| # | Decision | Why (evidence) |
|---|---|---|
| D1 | **One repository** created from the LabConstrictor template. Project name **StarDist2D**, which gives the package `stardist2d` and the bridge module `stardist2d_lc_tools`. | The template already builds, bundles, installs and registers `src/<pkg>` + `src/<pkg>_lc_tools` (`setup.py`, `bump_constructor.py:238`, `post_install.*`). The package name is derived from the project name. A project named "StarDist" would give `stardist` and hide the real StarDist library. |
| D2 | **Python 3.12, TF 2.20 + tf-keras 2.20 (Keras 2), stardist 0.9.2, csbdeep 0.8.2, numpy 2.1.3, bioimageio.core 0.5.11.** This is the same stack as ZeroCostDL4Mic 1.20.8 and DL4MicEverywhere. | Models and results stay interchangeable across the three projects. Keras 3 is ruled out because csbdeep raises `NotImplementedError` in `export_SavedModel` under Keras 3, which breaks the BioImage.IO export (csbdeep `utils/tf.py:227`). The draft requirements resolve with wheels only on win_amd64, macOS arm64 and Linux x86_64 (uv, 2026-11-11). |
| D3 | **Platforms: Windows x86_64 (CPU), macOS Apple Silicon (CPU), Linux x86_64 (CPU or NVIDIA GPU).** Intel Mac is dropped. | TF has published no Intel-Mac wheels since 2.16.2, and a second TF pin per platform is impossible in the template (one pin per package name, `merge_requirements.py:156-170`). |
| D4 | **No Windows GPU.** Train on Linux, Colab or DL4MicEverywhere; predict anywhere. | The TF 2.10.1 route fails: the app has one Python version, zarr 3.4/numcodecs need Python ≥3.12 and numpy 2, and TF 2.10 needs Python ≤3.10 and numpy 1.x. numpy cannot be per-OS because the main lock installs it first. DL4MicEverywhere gets Windows GPU only through Docker + WSL2, and native Fiji/napari/QuPath cannot reach that. |
| D5 | **Fix the template first (blocking).** Limit `tensorflow[and-cuda]` in the GPU file to Linux. | **Verified bug:** on any Windows PC with an NVIDIA driver the installer picks `requirements_gpu`, whose universal lock pins `nvidia-nccl-cu12` with no marker. That package has Linux-only wheels, so `pip install` fails and the install stops (`post_install.bat:58-79`). With markers (`tensorflow[and-cuda]==X ; sys_platform=="linux"` + `tensorflow==X ; sys_platform!="linux"`), uv marks every `nvidia-*` line Linux-only. Tested. |
| D6 | Logic in `src/stardist2d`; the notebook is text + one-line widget steps; the bridge is a thin layer. MorphoBricks rules enforced by `tools/check.py`. | Requested standard. One code path for notebook, CLI and hosts. |
| D7 | Hosts in v1: **Segment (fluorescence), Segment (H&E/RGB), Segment a folder, Quality control, Check installation.** Training stays in the notebook and CLI. | No job concept in the bridge; Fiji starts a new worker every run; a host kills a tool 3 s after Cancel. |

---

## 1. Value check: why this app, given the existing StarDist plugins

- **Fiji:** the Fiji StarDist plugin needs a model exported with TF 1.x (csbdeep `tf.py:227-235`). Models trained with ZeroCostDL4Mic on TF2 cannot be exported for it. **The bridge is the only way to run a ZeroCost-trained model in Fiji.** This is the main value.
- **QuPath:** we run exactly the same Python prediction as the notebook (parity), with region runs. Its own StarDist extension needs exported models (TF2 SavedModel compatibility not checked).
- **napari:** stardist-napari already loads model folders, so there is little extra value here beyond one install and the same thresholds and normalisation.
- **Notebook:** a local, installer-based ZeroCostDL4Mic StarDist 2D that also runs on Colab.

---

## 2. Repository layout

```
StarDist2D/                                   ← from the LabConstrictor template
├── notebooks/StarDist_2D/StarDist_2D_ZeroCostDL4Mic.ipynb
├── notebooks/StarDist_2D/requirements.yaml   (draft provided)
├── src/stardist2d/                           science package (MorphoBricks layers)
├── src/stardist2d_lc_tools/__init__.py       @tool declarations only (draft provided)
├── lc_tests/cases.json + fixtures/           labconstrictor-tools test cases
├── tests/                                    OUR pytest suite (template tests deleted)
├── tools/code_rules.py (byte-identical, SHA-tested), tools/check.py, tools/code_rules_baseline.json
├── .github/workflows/stardist2d_tests.yml    (template's tests.yml name is taken)
├── labconstrictor-app.yaml                   (draft provided, schema-valid)
├── pyproject.toml, AGENTS.md, CHANGELOG.md, docs/
```

### 2.1 Package `stardist2d`

Every module has a layer; the set of modules must equal the set of layer keys, `notebook/__init__` included.

| Module | Layer | From cell | Notes |
|---|---|---|---|
| `__init__.py` | — | 8 | Sets `os.environ.setdefault("TF_USE_LEGACY_KERAS","1")` **before any TF import**. No host sets it, and csbdeep reads it when it is imported. Lightweight: the installer's `import stardist2d` must succeed. |
| `config.py` | pure | 20, 28 | Frozen settings dataclasses. Patch-size and step rules. ≤8 arguments per function. |
| `augment.py` | pure | 23 | |
| `metrics.py` | pure | 36 | Binary IoU, `stardist.matching` row with argument order fixed. |
| `objects.py` | pure | 41 | Centroids, counts, polygons → GeoJSON/ROI from **StarDist `details['coord']`**, as the notebook's ROI zips do. |
| `data.py` | io | 17, 20, 28 | Image/mask pairing, alpha drop, seeded split, test dataset. Refuses empty or relative paths. |
| `pretrained.py` | io | 25 | Both sources: `from_pretrained` (GitHub release → `~/.keras`) **and** the notebook's cloud.mpi-cbg.de / BioImage Model Zoo / local-folder options. |
| `model.py` | orchestration | 28 | Lazy TF import. |
| `train.py` | orchestration | 30 | Keras callback (progress / cancel hooks for the CLI and future tools). |
| `predict.py` | orchestration | 41, 43 | `predict_instances` via `_predict_instances_generator` (tile by tile, so `check_cancel`/`progress` work). Stacks are resumable. Big images use zarr. |
| `qc.py` | orchestration | 36 | |
| `export.py` | io | 38, 41, 43 | Label TIFF, ImageJ ROI zip, CSVs, BioImage.IO (tf-keras route). |
| `report.py` | io | 8 | PDFs built from explicit arguments, no globals. |
| `plotting.py` | ui | 20, 34, 36, 41 | Functions return figures. |
| `environment.py` | io | 8, 11 | GPU/TF info, requirements freeze, version check. No magics. |
| `provenance.py` | io | new | `run_config.json` beside every output; atomic writes. **Do not name a function `source_fingerprint`** (reserved name). |
| `bridge.py` | orchestration | new | What `_lc_tools` calls: checks inputs (RGB must be `shape[-1]==3` and uint8; refuses >65,535 objects with a "use a region" message), adds the region offset, builds outputs. |
| `notebook/steps.py`, `notebook/widgets.py` | ui | all forms | One step per original code cell, same parameter names and defaults. Must work on ipywidgets 7.7 (Colab) and 8.1 (local). DL4MicEverywhere's `notebook_autoconversion` is a reference for the form→widget mapping. |
| `cli.py` | orchestration | new | `stardist2d train / qc / predict / export`. |

Names reserved by other bricks: the real `RESERVED_NAMES` list (`code_rules.py:209-245`) is exact names, and `normalize` is **not** in it. Avoid `source_fingerprint`, `check_labels`, `read_mask`, `find_masks`, `percentile_range`, `min_max`, `named_axes`, `check_sources`.

### 2.2 Notebook

- **Text:** all 31 markdown cells copied word for word, sections 0–7. Colab-only passages get "Colab users / JupyterLab users" callouts. Every change is listed in the CHANGELOG.
- **Version cell:** the template snippet. `current_version` must be a literal string. Use **independent semver `1.0.0`**, and record "port of ZeroCostDL4Mic StarDist 2D 1.20.8" in the notebook, README and CHANGELOG. (`1.20.8-lc.1` is not valid PEP 440.)
- **Install cell:** inside `if "google.colab" in sys.modules:`, a `subprocess` pip install of the pinned stack + `git+https://github.com/<org>/StarDist2D@v<tag>`, then a restart. Never install ipywidgets 8 on Colab. `enable_custom_widget_manager()` also goes inside the guard.
- **Other cells:** Drive mount and `nvidia-smi` are Colab-guarded. Every other code cell is `# @title` + one call. No `!`/`%` magics, no `os.chdir`.
- **MorphoBricks / LabConstrictor overlap:** the template's version snippet imports ipywidgets, yaml and requests. That clashes with CellPatcher's stdlib-only bootstrap test. Our notebook test exempts the template's version cell by name (documented decision).

### 2.3 Bridge (draft, `labconstrictor-tools check` passes)

- All tools have ≤8 parameters. Tiling is automatic (the `n_tiles` parameter was removed to meet the limit).
- Outputs:
  - `LabelsOut` (uint16; refused above 65,535 objects);
  - `ShapesOut` from StarDist polygons with `label` and `prob` properties (QuPath turns numeric properties into measurements);
  - `TableOut`;
  - `MessageOut`.
- Not in v1: a probability-map `ImageOut` (cheap to add later).
- Known host limits, documented for users:
  - QuPath makes annotations, not detections, and does not link tables to objects;
  - Fiji reloads TensorFlow and the model on every run;
  - Cancel kills the worker during the final non-maximum suppression step;
  - `Axes("YXS")` only checks the number of dimensions, so the bridge validates the RGB layout itself.

---

## 3. Phases

| Phase | Content | Exit criterion |
|---|---|---|
| **P−1 (blocking, ½ day)** | LabConstrictor template PR: (1) the GPU file limits `tensorflow[and-cuda]` to Linux; (2) the version-cell snippet reads the nested `notebook_latest_versions.yaml`; (3) optional: the Welcome updater re-runs `register` after updating `src/`. | uv lock of the GPU file has every `nvidia-*` line marked `sys_platform == 'linux'`; template tests pass. |
| **P0 (½ day)** | Create the repo from the template. Run the initialisation form with name `StarDist2D`. Delete the template's `tests/`. Add pyproject (ruff 0.17.3, MorphoBricks rules, `[tool.morphobricks.code-rules]`), byte-identical `code_rules.py` + SHA test, `check.py`, empty baseline, AGENTS.md, CHANGELOG, and the `stardist2d_tests.yml` workflow. That workflow installs the lock on Linux, runs `check.py`, pytest, `import stardist2d`, `labconstrictor-tools check` and `test`. | CI green on an empty package. |
| **P1 (3–4 days)** | Write the package (§2.1) with synthetic-data tests. **Parity harness:** run the original 1.20.8 code on a tiny dataset with a fixed seed and record labels, metrics, CSVs and ROI vertices as golden files. Our code must match, except for listed intentional bug fixes. TF tests are marked `slow`. | Parity tests green; ≤600 lines per file, ≤60 lines per function, ≤8 parameters; baseline empty. |
| **P2 (2 days)** | Write `notebook/` steps + the notebook. Notebook tests: nbformat, no outputs, `# @title`, no magics, Colab guards, version index; an automatic checklist that every original `#@param` name and default exists; headless widget clicks; nbclient smoke run (tiny data, 1 epoch, CPU). | Runs locally (JupyterLab 4, ipywidgets 8.1) and on Colab (7.7). Manual check on Colab. |
| **P3 (2 days)** | Write `_lc_tools` + `bridge.py` + `lc_tests/cases.json`. Cases cover shape, dtype, `n_labels`, `max_seconds`, `progress_events_min`, plus a `--check-cancel` case on an image large enough to run several seconds on CPU. | `labconstrictor-tools test` green in CI. |
| **P4 (1 day)** | Write `requirements.yaml` (draft provided), the manifest and the platforms. Build the installers. | Install + `import` + register on Windows (CPU, **including a PC with an NVIDIA driver**), macOS arm64, Linux CPU, Linux NVIDIA. |
| **P5 (1–2 days)** | Human tests in Fiji, napari and QuPath: segment, region run, Cancel, H&E image, a custom ZeroCost-trained model folder, a model trained on Colab or DL4MicEverywhere loaded locally. | Results CSV filled in; any gaps go upstream as host issues. |
| **P6** | README, user guide, CHANGELOG, release. Tag, `RELEASE_TAG`, `current_version` and the version index move together. | Release installers published. |

---

## 4. Risks that remain

1. **The tf-keras route is "not regularly tested" upstream** (csbdeep's own warning). It is the route ZeroCostDL4Mic and DL4MicEverywhere ship, and P1 parity tests guard it.
2. **bioimageio.core 0.5.11** (legacy API) with numpy 2 and current xarray has not been checked. The P1 export test catches it.
3. **CPU-only training on Windows and macOS is slow.** The docs point users to Colab or DL4MicEverywhere for training.
4. **LabConstrictor-Tools is alpha** (v0.3.0-alpha.1). The Windows `.bat` hooks have never been run by a person.
5. **Weights for the notebook's "from Stardist Fiji" choices** come from cloud.mpi-cbg.de. We could not reach that server from here, so equivalence with the GitHub-release weights is not checked. Both sources stay available.
6. **The template's P−1 PR** must be merged (or the app pinned to a fixed template version) before the first installer.

## 5. Decisions taken (2026-11-11)

- **Bugs in 1.20.8:** fix them. Every intentional difference from 1.20.8 is listed in the CHANGELOG, with a test showing the old and the new behaviour.
- **Version:** independent `1.0.0`, recorded as a port of ZeroCostDL4Mic StarDist 2D 1.20.8.
- **Template fix (P−1):** done. LabConstrictor PR #39 (template 0.1.22: GPU file + version cell).
- **QuPath:** annotations are acceptable for v1. A detections PR to LabConstrictor-QuPath is planned for later.
- **Platforms:** Windows CPU, macOS Apple Silicon CPU, Linux CPU/NVIDIA GPU. Intel Mac is dropped.
- **Training** stays in the notebook and CLI only for v1.
- **Repository:** `CellMigrationLab/StarDist2D`, with package `stardist2d` and bridge module `stardist2d_lc_tools`.
- **Next:** P0 + P1 in the new repository, once it is created and added to the session.

## 6. Open scientific question found during P1

1.20.8 normalises multi-channel (e.g. H&E) images **per channel** for training (cell 28, `axis_norm=(0,1)`) but
**jointly** for quality control and prediction (cells 36, 41, `axis_norm=(0,1,2)`). The port keeps both as they are
(`stardist2d.prepare.training_axes` / `prediction_axes`) because changing either changes results.
*Decided (November 2026): normalise channels jointly everywhere (`prepare.normalization_axes`), listed in the CHANGELOG.*
