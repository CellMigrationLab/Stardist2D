# Notebook screenshots (JupyterLab, local run)

Taken on 2026-11-11 by `run_notebook.js` (Playwright) driving the notebook in JupyterLab 4.4.0 exactly as a person
would: run all cells, fill each form, click its button. Linux, CPU only (no GPU), Python 3.12, TensorFlow 2.20 +
tf-keras 2.20, StarDist 0.9.2, ipywidgets 8.1.7.

The run uses the example dataset shipped with the app (section 3.1, offline copy), starts from StarDist's pretrained
`2D_versatile_fluo` (section 3.4, `Model_from_file`), and trains 3 epochs of 10 steps on 256×256 patches
(about 16 s on the CPU). Quality control on the two test images: F1 0.96–0.97, IoU 0.82–0.84.

| File | Section |
|---|---|
| `01_notebook_opened.png` | the notebook as it opens |
| `02_download_dataset.png` | 3.1 example dataset |
| `03_training_parameters.png` | 3.2 training parameters and a training pair |
| `04_augmentation.png` | 3.3 augmentation |
| `05a_pretrained.png` | 3.4 pretrained weights |
| `05_prepare.png` | 4.1 data and model |
| `06_training.png` | 4.2 training |
| `07_loss_curves.png` | 5.1 loss curves |
| `08_quality_control.png` | 5.2 metrics table and QC viewer |
| `09_bioimage_export.png` | 5.3 BioImage Model Zoo export |
| `10_predictions.png` | 6.1 predictions with the overlay |

To run it again: start JupyterLab on port 8899 in a folder holding the notebook (with `src/` on `PYTHONPATH`, the
notebook setting `windowingMode` set to `none` so every cell is rendered), copy the pretrained model folder to
`~/StarDist2D/pretrained/2D_versatile_fluo`, then `OUT=<folder> node run_notebook.js`.

## napari (LabConstrictor-napari plugin)

`napari/napari_01_form.png` and `napari/napari_02_result.png`: napari 0.10.0 with the napari-labconstrictor plugin
(from the LabConstrictor-napari repository) in its own environment. The StarDist2D app is registered exactly as the
installer does (`labconstrictor_tools register ... --app-manifest labconstrictor-app.yaml`, app Python = the
StarDist2D environment). "StarDist 2D: Segment nuclei (fluorescence)" on the example test image with
`2D_versatile_fluo`: a Labels layer, a Shapes layer (284 outlines) and the per-object table, in 5 s on the CPU.
Run by `napari/run_napari.py <folder with v312/ (app env) and napari/ (host env)>` under `xvfb-run`.
