# StarDist2D

Segment star-convex nuclei and cells in 2D images with [StarDist](https://github.com/stardist/stardist): train your own
model, check its quality, export it to the BioImage Model Zoo and predict, from a notebook (locally or on Google Colab)
or directly in **Fiji, napari and QuPath**.

StarDist2D is the [LabConstrictor](https://github.com/CellMigrationLab/LabConstrictor) version of the
[ZeroCostDL4Mic](https://github.com/HenriquesLab/ZeroCostDL4Mic) StarDist 2D notebook (v1.20.8): the same text, the
same steps and parameters, the same output files, with the code moved into the `stardist2d` Python package. What
changed, and every bug fixed on the way, is listed in the [CHANGELOG](CHANGELOG.md).

![Quality control in the notebook](docs/screenshots/08_quality_control.png)

## How to Download and Install StarDist2D?

The installer would only be available once StarDist2D developer has created a release.
Please, go to [this page](.tools/docs/download_executable.md) and check if the installing instructions are there.
In case there is no documentation there, please [create an issue](https://github.com/CellMigrationLab/Stardist2D/issues) asking the developer to create a release.

Installers: Windows x86_64, macOS Apple Silicon, Linux x86_64. Training uses an NVIDIA GPU on Linux; on Windows and
macOS it runs on the CPU (slow), so train on Colab or [DL4MicEverywhere](https://github.com/HenriquesLab/DL4MicEverywhere)
if you need speed. A model trained anywhere works everywhere for prediction.

## Use it

- **Notebook** (from the app's Welcome page, or on [Colab](https://colab.research.google.com/github/CellMigrationLab/Stardist2D/blob/main/notebooks/StarDist_2D/StarDist_2D_ZeroCostDL4Mic.ipynb)):
  follow sections 1 to 6. Section 3.1 gives a small example dataset even offline (6 training pairs and 2 test pairs
  from [Zenodo 7949940](https://doi.org/10.5281/zenodo.7949940)). Screenshots of a full run: [docs/screenshots](docs/screenshots/README.md).
- **Fiji, napari, QuPath** (with the LabConstrictor plugin of each): pick *StarDist2D* and a tool.

  | Tool | What you get |
  |---|---|
  | Segment nuclei (fluorescence) | label image, outlines (ROI Manager / Shapes layer / annotations), per-object table; channel choice and region runs |
  | Segment nuclei (H&E / RGB) | the same for RGB images with `2D_versatile_he` or your 3-channel model |
  | Segment a folder | the notebook's section 6.1 outputs for every image of a folder |
  | Quality control | your model's prediction compared with a ground-truth mask |
  | Check this installation | memory, disk, GPU and TensorFlow of the app |

  Models: `2D_versatile_fluo`, `2D_versatile_he`, `2D_paper_dsb2018` (downloaded once), or the folder of a model you
  trained in the notebook.
- **Command line**: `python -m stardist2d train|qc|predict ...` (see `python -m stardist2d --help`).

## Credits

**Please cite StarDist when you use this app:** Schmidt U., Weigert M., Broaddus C., Myers G. *Cell Detection with
Star-Convex Polygons*. MICCAI 2018, Lecture Notes in Computer Science 11071, pp. 265–273,
[doi:10.1007/978-3-030-00934-2_30](https://doi.org/10.1007/978-3-030-00934-2_30) (preprint:
[arXiv:1806.03535](https://arxiv.org/abs/1806.03535)).

Please also cite ZeroCostDL4Mic, the notebook this app is built on: von Chamier L., Laine R. F. et al. *Democratising
deep learning for microscopy with ZeroCostDL4Mic*. Nature Communications 12, 2276 (2021),
[doi:10.1038/s41467-021-22518-0](https://doi.org/10.1038/s41467-021-22518-0); and LabConstrictor, which packages it:
Hidalgo-Cenalmor I. et al., [doi:10.48550/arXiv.2603.10704](https://doi.org/10.48550/arXiv.2603.10704).

The installer images use the StarDist logo (BSD 3-Clause, see [app/logo](app/logo/README.md)).

## Documentation for StarDist2D's Developers

- How the code is organised and the rules it follows (MorphoBricks code standard): [AGENTS.md](AGENTS.md).
- Checks before a pull request: `python tools/check.py` (tests, shared code rules, Ruff). The notebook is generated:
  `python tools/build_notebook.py`. The Fiji / napari / QuPath tools: `labconstrictor-tools test --module stardist2d_lc_tools --pythonpath src --cases lc_tests/cases.json`.
- The plan and its decisions: [docs/plan.md](docs/plan.md).
- Internal documentation on how to upload notebooks or create executables is available in the [.tools/docs](.tools/docs/README.md) folder.
