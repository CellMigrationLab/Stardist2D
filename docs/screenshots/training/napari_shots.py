"""Screenshots of training in napari: the form, training in progress, results, segmenting with the new model, QC."""
import sys, time
from pathlib import Path
import napari, tifffile
from qtpy.QtWidgets import QApplication
OUT = Path(sys.argv[1]); DATA = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
viewer = napari.Viewer(show=True); viewer.window._qt_window.resize(1600, 1000)
qa = QApplication.instance()
_, w = viewer.window.add_plugin_dock_widget("napari-labconstrictor", "LabConstrictor tools")
def pump(sec=0.3):
    end = time.time() + sec
    while time.time() < end: qa.processEvents(); time.sleep(0.02)
def shot(name): pump(0.8); viewer.screenshot(str(OUT / f"{name}.png"), canvas_only=False); print("SHOT", name, "|", w.status.text()[:120])
def start(tool, values):
    w.tool_box.value = tool; pump(1.5)
    for k, v in values.items(): getattr(w.gui, k).value = v
    pump(0.5)
def run_and_wait(progress_shot=None, timeout=3000):
    w.last_task = None; w.gui(); end = time.time() + timeout; taken = False
    while not (w.last_task is not None and w.task is w.last_task) and time.time() < end:
        pump(0.5)
        if progress_shot and not taken and "epoch 2/" in w.status.text():
            shot(progress_shot); taken = True
    pump(2)
pump(1); w.app_box.value = "StarDist2D"; pump(0.5)
models = OUT / "models"; models.mkdir(exist_ok=True)
image = tifffile.imread(DATA / "Test - Images" / "test_01.tif")
viewer.add_image(image, name="test_01")
start("StarDist 2D: Train a model", {"training_images": DATA / "Training - Images", "training_masks": DATA / "Training - Masks",
      "model_folder": models, "model_name": "my_nuclei", "number_of_epochs": 3, "start_from": "2D_versatile_fluo"})
shot("napari_1_train_form")
run_and_wait("napari_2_training_progress")
shot("napari_3_training_results")
start("StarDist 2D: Segment nuclei (fluorescence)", {"model": "Custom model folder", "model_folder": models / "my_nuclei"})
run_and_wait(); shot("napari_4_segment_with_trained_model")
start("StarDist 2D: Quality control on a folder", {"model_folder": models / "my_nuclei", "test_images": DATA / "Test - Images", "test_masks": DATA / "Test - Masks"})
run_and_wait(); shot("napari_5_qc_on_a_folder")
viewer.close()
