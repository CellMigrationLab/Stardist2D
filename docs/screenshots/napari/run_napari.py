"""Run the StarDist2D tools inside napari (off-screen) as an installed LabConstrictor app, and take screenshots."""
import os, sys, subprocess, tempfile, time, json
from pathlib import Path
S = Path(sys.argv[1]); REPO = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[3]
home = Path(tempfile.mkdtemp(prefix="lc_home_")); os.environ["LC_HOME"] = str(home)
app_python = str(S / "v312/bin/python")
reg = subprocess.run([app_python, "-m", "labconstrictor_tools", "register", "--name", "StarDist2D", "--prefix", str(S / "v312"),
                      "--module", "stardist2d_lc_tools", "--pythonpath", str(REPO / "src"), "--version", "1.0.0",
                      "--display-name", "StarDist2D", "--app-manifest", str(REPO / "labconstrictor-app.yaml")],
                     capture_output=True, text=True)
print("register:", reg.returncode, reg.stdout[-300:], reg.stderr[-300:])
import napari, tifffile, numpy as np
from qtpy.QtWidgets import QApplication
viewer = napari.Viewer(show=True); viewer.window._qt_window.resize(1500, 900)
qa = QApplication.instance()
_, w = viewer.window.add_plugin_dock_widget("napari-labconstrictor", "LabConstrictor tools")
def pump(sec=0.3):
    end = time.time() + sec
    while time.time() < end: qa.processEvents(); time.sleep(0.02)
def until(cond, timeout=600):
    end = time.time() + timeout
    while not cond() and time.time() < end: qa.processEvents(); time.sleep(0.05)
    pump(0.5)
pump(1)
print("apps:", list(w.app_box.choices))
image = tifffile.imread(REPO / "example_data/StarDist_2D/Test - Images/test_01.tif")
viewer.add_image(image, name="test_01", contrast_limits=(float(np.percentile(image, 1)), float(np.percentile(image, 99.8))))
w.app_box.value = "StarDist2D"; pump(0.5)
print("tools:", list(w.tool_box.choices))
w.tool_box.value = "StarDist 2D: Segment nuclei (fluorescence)"; pump(1.5)
viewer.screenshot(str(S / "napari_run/napari_01_form.png"), canvas_only=False)
w.last_task = None; w.gui()
until(lambda: w.last_task is not None and w.task is w.last_task, 900)
pump(2)
print("layers:", [(l.name, type(l).__name__) for l in viewer.layers])
print("status:", w.status.text()[:300])
viewer.screenshot(str(S / "napari_run/napari_02_result.png"), canvas_only=False)
labels = [l for l in viewer.layers if type(l).__name__ == "Labels"]
shapes = [l for l in viewer.layers if type(l).__name__ == "Shapes"]
result = {"labels_layers": len(labels), "shapes_layers": len(shapes),
          "n_labels": int(labels[0].data.max()) if labels else 0, "n_shapes": len(shapes[0].data) if shapes else 0}
print(json.dumps(result))
viewer.close()
