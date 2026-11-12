
// ================================================================== GUI test body: the StarDist2D app's Segment tool in QuPath
import javafx.embed.swing.SwingFXUtils
import javax.imageio.ImageIO
import java.util.concurrent.CountDownLatch
import java.awt.image.BufferedImage
import javafx.stage.Window

def OUT = new File(System.getProperty("lc.test.out", "/tmp/qp_shots")); OUT.mkdirs()
def FIXTURES = System.getProperty("lc.test.fixtures", new File("tests/fixtures").absolutePath)
def results = [:]
def fx = { Closure c ->
    if (Platform.isFxApplicationThread()) return c()
    def latch = new CountDownLatch(1); def box = [null, null]
    Platform.runLater { try { box[0] = c() } catch (Throwable t) { box[1] = t } finally { latch.countDown() } }
    latch.await()
    if (box[1] != null) throw box[1]
    return box[0]
}
def expect = { String name, boolean ok, Object detail = "" ->
    results[name] = ok
    println((ok ? "PASS " : "FAIL ") + name + (ok ? "" : " -> " + detail))
}
def shot = { String name, Stage st ->
    Thread.sleep(800)
    fx { ImageIO.write(SwingFXUtils.fromFXImage(st.scene.snapshot(null), null), "png", new File(OUT, name + ".png")) }
    println "SHOT " + name
}
def waitIdle = { dlg, int seconds = 600 ->
    def end = System.currentTimeMillis() + seconds * 1000
    while (System.currentTimeMillis() < end) { Thread.sleep(250); if (!dlg.running && fx { !dlg.runButton.disable }) return true }
    return false
}
def status = { dlg -> fx { dlg.status.text } }
def resultStages = { Window.getWindows().findAll { it instanceof Stage && it.showing && it.title?.contains(": ") && it.title != "LabConstrictor: last run" } }
def DATA = System.getProperty("lc.test.data")
def MODELS = System.getProperty("lc.test.models")

Thread.start("lc-gui-test") {
    try {
        fx {
            def dir = new File("/tmp/qp_project_stardist2d_train"); dir.deleteDir()
            qupath.lib.gui.prefs.PathPrefs.imageTypeSettingProperty().set(qupath.lib.gui.prefs.PathPrefs.ImageTypeSetting.NONE)
            def project = qupath.lib.projects.Projects.createProject(dir, BufferedImage)
            def server = qupath.lib.images.servers.ImageServers.buildServer(FIXTURES + "/stardist_test_01.tif")
            def entry = project.addImage(server.getBuilder()); entry.setImageName("stardist_test_01.tif")
            project.syncChanges(); qupath.setProject(project); qupath.openImageEntry(entry); entry
        }
        Thread.sleep(3000)
        def dlg = fx { lcShow() }
        Thread.sleep(1000)
        fx { dlg.appBox.value = "StarDist2D" }; Thread.sleep(500)
        expect("seven_tools", fx { (dlg.toolBox.items as List).size() } == 7, fx { dlg.toolBox.items as List })
        fx { dlg.toolBox.value = "StarDist 2D: Train a model" }; Thread.sleep(1500)
        fx {
            dlg.setters["training_images"](DATA + "/Training - Images")
            dlg.setters["training_masks"](DATA + "/Training - Masks")
            dlg.setters["model_folder"](MODELS)
            dlg.setters["model_name"]("qupath_nuclei")
            dlg.setters["number_of_epochs"](3)
            dlg.setters["start_from"]("2D_versatile_fluo")
        }
        Thread.sleep(800)
        shot("qp_train_1_form", dlg.stage)
        fx { dlg.run() }
        def end = System.currentTimeMillis() + 1500 * 1000; boolean progressShot = false
        while (System.currentTimeMillis() < end) {
            Thread.sleep(500)
            if (!progressShot && status(dlg).contains("epoch 2/")) { shot("qp_train_2_progress", dlg.stage); progressShot = true }
            if (!dlg.running && fx { !dlg.runButton.disable }) break
        }
        println "STATUS " + status(dlg)
        expect("progress_per_epoch_shown", progressShot, status(dlg))
        expect("training_done", status(dlg).startsWith("done") || status(dlg).contains("✔"), status(dlg))
        expect("model_written", new File(MODELS + "/qupath_nuclei/weights_best.h5").isFile() && new File(MODELS + "/qupath_nuclei/thresholds.json").isFile(), MODELS)
        shot("qp_train_3_dialog_after", dlg.stage)
        resultStages().eachWithIndex { st, i -> shot("qp_train_4_results_" + i, st) }
        fx { dlg.toolBox.value = "StarDist 2D: Segment nuclei (fluorescence)" }; Thread.sleep(1500)
        fx { dlg.setters["model"]("Custom model folder"); dlg.checks["model_folder"].selected = true; dlg.setters["model_folder"](MODELS + "/qupath_nuclei") }
        Thread.sleep(500)
        fx { dlg.run() }; Thread.sleep(2000)
        expect("segment_finished", waitIdle(dlg, 600), status(dlg))
        def found = fx { qupath.imageData.hierarchy.annotationObjects.findAll { it.metadata["labconstrictor.output"] == "stardist_outlines" } }
        println "OUTLINES " + found.size()
        expect("outlines_from_the_trained_model", found.size() > 100, found.size())
        shot("qp_train_5_segment_dialog", dlg.stage)
        shot("qp_train_6_viewer", qupath.stage)
    } catch (Throwable t) { t.printStackTrace(); results["EXCEPTION"] = false }
    finally { println "RESULTS " + results; Thread.sleep(500); System.exit(results && results.every { it.value } ? 0 : 1) }
}
