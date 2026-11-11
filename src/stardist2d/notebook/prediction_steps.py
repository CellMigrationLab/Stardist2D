"""Notebook section 6: predictions on unseen images, and on very large images (1.20.8 cells 41 and 43)."""

from __future__ import annotations

from typing import Any

from IPython.display import display

from stardist2d import batch, model, plotting, predict, prepare
from stardist2d.notebook.forms import Field, Form, message
from stardist2d.notebook.session import Session


def predictions(session: Session) -> Form:
    """Section 6.1: predict a folder of single images or stacks and write labels, ROIs and counts (cell 41)."""

    def run(values: dict[str, Any]) -> None:
        folder = session.model_from_choice(values["Use_the_current_trained_model"], values["Prediction_model_folder"])
        message(f"The {folder.name} network will be used.")
        star_dist = model.load_model_folder(folder)
        images, results = session.path(values["Data_folder"]), session.path(values["Results_folder"])
        rois = values["Region_of_interests"]
        hooks = predict.Hooks(progress=lambda fraction, text: print(f"{text}", flush=True))
        if values["Data_type"] == "Stacks":
            outcome = batch.predict_stacks(star_dist, images, results, rois, hooks)
            for name in outcome.skipped:
                message(f"The stack {name} has already been processed: remove its result folder to redo it.", True)
            for name, counts in outcome.counts.items():
                display(
                    plotting.objects_per_frame(
                        counts, results / f"{name.rsplit('.', 1)[0]}_Object_detected_vs_frame_number.png"
                    )
                )
        else:
            outcome = predict.predict_folder(star_dist, images, results, export_rois=rois, hooks=hooks)
            image = prepare.normalize_image(outcome.last_image, prepare.prediction_axes(outcome.last_image))
            stem = outcome.counts[-1][0].rsplit(".", 1)[0]
            print("One example image is displayed below:")
            display(plotting.overlay(image, outcome.last_labels, results / f"{stem}_overlay.tif"))
        message("Predictions completed")

    fields = [
        Field("Data_folder", ""),
        Field("Results_folder", ""),
        Field("Data_type", "Single_Images", "choice", ("Single_Images", "Stacks")),
        Field("Region_of_interests", True, "bool", help="Also write ImageJ ROI zips (Fiji ROI Manager)"),
        Field("Use_the_current_trained_model", True, "bool"),
        Field("Prediction_model_folder", ""),
    ]
    return Form("Generate prediction(s) from unseen dataset", fields, run, "Predict").show()


def big_data_predictions(session: Session) -> Form:
    """Section 6.2: predict very large images block by block (cell 43)."""

    def run(values: dict[str, Any]) -> None:
        folder = session.model_from_choice(values["Use_the_current_trained_model"], values["Prediction_model_folder"])
        if values["block_size_Y"] != values["block_size_X"]:
            raise ValueError("block_size_Y and block_size_X must be equal in this version")
        settings = batch.BigImageSettings(
            block_size=values["block_size_Y"],
            min_overlap=values["min_overlap"],
            n_tiles=(values["n_tiles_Y"], values["n_tiles_X"]),
            is_stack=values["Is_stack"],
            mask_images=values["Mask_images"],
            region_of_interests=values["Region_of_interests"],
        )
        done = batch.predict_big_folder(
            model.load_model_folder(folder),
            session.path(values["Data_folder"]),
            session.path(values["Results_folder"]),
            settings,
        )
        message(f"Predictions completed for {len(done)} image(s)")

    fields = [
        Field("Data_folder", ""),
        Field("Results_folder", ""),
        Field("Use_the_current_trained_model", True, "bool"),
        Field("Prediction_model_folder", ""),
        Field("block_size_Y", 512, "int"),
        Field("block_size_X", 512, "int"),
        Field("min_overlap", 50, "int"),
        Field("n_tiles_Y", 1, "int"),
        Field("n_tiles_X", 1, "int"),
        Field("Is_stack", True, "bool"),
        Field("Mask_images", True, "bool"),
        Field("Region_of_interests", True, "bool"),
    ]
    return Form("Generate prediction(s) from unseen dataset (Big data)", fields, run, "Predict").show()
