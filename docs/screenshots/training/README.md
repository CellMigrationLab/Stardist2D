# Training from napari, Fiji and QuPath (StarDist2D 1.1.0)

Real runs, 12 November 2026, on a virtual screen (xvfb), CPU only. The StarDist2D app is registered the way the
installer registers it (`labconstrictor-tools register --app-manifest labconstrictor-app.yaml`), with the code of
the 1.1.0 branch. Every host ran the same task:

1. **Train a model** on the bundled example dataset (`example_data/StarDist_2D`: 6 training pairs, 1024×1024),
   starting from `2D_versatile_fluo`, 3 epochs, the notebook's default advanced parameters.
2. **Segment nuclei (fluorescence)** on `Test - Images/test_01.tif` with *Custom model folder* = the new model.

| Host | Training | Validation (1 image kept by the 10 % split) | Segmentation with the new model |
|---|---|---|---|
| napari 0.10 (napari-labconstrictor) | 1 min 50 s; progress bar and "epoch 2/3: loss …, val_loss …" | F1 0.977 | 197 outlines (Shapes layer), labels, table |
| Fiji (LabConstrictor-Fiji script) | 2 min 16 s; status bar "LabConstrictor: epoch 2/3 … (press Esc to cancel)" | F1 0.980; IoU 0.890 | 193 ROIs (ROI Manager and overlay), 16-bit labels, table |
| QuPath 0.7.0 (LabConstrictor-QuPath, with the output-ownership fix) | 1 min 56 s; progress bar and "epoch 2/3 …" | IoU 0.891, 170 true positives | 199 annotations |

The numbers differ slightly between hosts because training is random (augmentation, patch sampling).

## Screenshots

- **napari:**
  - [form](napari_1_train_form.png);
  - [training in progress](napari_2_training_progress.png);
  - [results](napari_3_training_results.png): loss table, validation QC, summary naming the model folder;
  - [segmenting with the trained model](napari_4_segment_with_trained_model.png);
  - [Quality control on a folder](napari_5_qc_on_a_folder.png).
- **Fiji:**
  - [form](fiji_1_train_form.png);
  - [start, with the CPU note](fiji_2a_training_starts_cpu_note.png);
  - [training in progress](fiji_2b_training_progress.png);
  - [results](fiji_3_training_results.png): Log with the summary, `training_curve` and `validation_quality` tables;
  - [segmenting with the trained model](fiji_4_segment_with_trained_model.png).
- **QuPath:**
  - [form](qupath_1_train_form.png);
  - [training in progress](qupath_2_training_progress.png);
  - [results window](qupath_3_training_results.png);
  - [segmenting with the trained model](qupath_4_segment_with_trained_model.png).

Cancel was tested separately in napari: a 500-epoch training cancelled after 25 s stopped at once. The model folder
kept `weights_best.h5` and `stardist2d_run.json` says "stopped after epoch 40 of 500; thresholds not optimised".

## How they were made

- **napari:** [`napari_shots.py`](napari_shots.py) drives the LabConstrictor widget (sets the form, presses Run,
  takes `viewer.screenshot` when the status shows "epoch 2/").
- **Fiji:** LabConstrictor-Fiji's harness (`tests/run_cases.py --real-apps`) with the cases
  [`fiji_case_train.json`](fiji_case_train.json) and [`fiji_case_segment_trained.json`](fiji_case_segment_trained.json).
  Two options were added to a local copy of the harness for these screenshots only:
  - `progress_every_s`, a full-screen capture every 15 s;
  - `arrange`, which lays out the windows side by side before the final capture.

  They are not part of LabConstrictor-Fiji. The progress images are cropped to the ImageJ main window.
- **QuPath:** the extension's runner (`tests/run_gui_test.sh`) with the test body
  [`qupath_gui_test_train_body.groovy`](qupath_gui_test_train_body.groovy) (6 checks, all passed).
