
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

Thread.start("lc-gui-test") {
    try {
        fx {
            def dir = new File("/tmp/qp_project_stardist2d"); dir.deleteDir()
            qupath.lib.gui.prefs.PathPrefs.imageTypeSettingProperty().set(qupath.lib.gui.prefs.PathPrefs.ImageTypeSetting.NONE)
            def project = qupath.lib.projects.Projects.createProject(dir, BufferedImage)
            def server = qupath.lib.images.servers.ImageServers.buildServer(FIXTURES + "/stardist_test_01.tif")
            def entry = project.addImage(server.getBuilder()); entry.setImageName("stardist_test_01.tif")
            project.syncChanges(); qupath.setProject(project); qupath.openImageEntry(entry); entry
        }
        Thread.sleep(3000)
        def dlg = fx { lcShow() }
        Thread.sleep(1000)
        expect("app_listed", fx { dlg.appBox.items as List }.contains("StarDist2D"), fx { dlg.appBox.items as List })
        fx { dlg.appBox.value = "StarDist2D" }; Thread.sleep(500)
        expect("five_tools", fx { (dlg.toolBox.items as List).size() } == 5, fx { dlg.toolBox.items as List })
        fx { dlg.toolBox.value = "StarDist 2D: Segment nuclei (fluorescence)" }; Thread.sleep(1500)
        shot("qp_stardist_1_form", dlg.stage)
        fx { dlg.run() }; Thread.sleep(2000)
        expect("run_finished", waitIdle(dlg, 600), status(dlg))
        println "STATUS " + status(dlg)
        def found = fx { qupath.imageData.hierarchy.annotationObjects.findAll { it.name?.startsWith("StarDist2D:stardist_outlines ") } }
        println "ANNOTATIONS " + found.size()
        expect("outlines_as_annotations", found.size() > 100, found.size())
        expect("measurements_from_properties", found.every { it.getMeasurementList().getMeasurementNames().containsAll(["label", "prob"]) }, found.take(3).collect { it.getMeasurementList().getMeasurementNames() })
        expect("status_done", status(dlg).startsWith("done") || status(dlg).contains("✔"), status(dlg))
        shot("qp_stardist_2_dialog_after", dlg.stage)
        fx { qupath.overlayOptions.showNamesProperty().set(false) }; Thread.sleep(800)
        shot("qp_stardist_3_viewer", qupath.stage)
    } catch (Throwable t) { t.printStackTrace(); results["EXCEPTION"] = false }
    finally { println "RESULTS " + results; Thread.sleep(500); System.exit(results && results.every { it.value } ? 0 : 1) }
}
