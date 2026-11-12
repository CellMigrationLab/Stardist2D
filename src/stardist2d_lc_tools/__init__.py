"""LabConstrictor tools bridge for StarDist 2D: what Fiji, napari and QuPath list.

DRAFT. This module only declares the tools and delegates to ``stardist2d.bridge``
(the science package). Following the LabConstrictor-Tools rules, only the standard
library and ``labconstrictor_tools`` are imported at module level; TensorFlow and
StarDist are imported inside ``stardist2d`` when a tool runs; ``stardist2d/__init__.py``
sets ``TF_USE_LEGACY_KERAS=1`` (tf-keras, as ZeroCostDL4Mic) before anything imports
TensorFlow, because no host sets it. Tiling is chosen automatically and prediction runs
tile by tile so that Cancel and progress work (``_predict_instances_generator``).
Outlines are StarDist's own polygons, as in the notebook's ImageJ ROI zips.
"""

from __future__ import annotations

from typing import Annotated, Literal

from labconstrictor_tools import (
    Advanced,
    ApplyTo,
    Axes,
    Description,
    EnabledWhen,
    Folder,
    Group,
    Image,
    Labels,
    LabelsOut,
    Max,
    MessageOut,
    Min,
    Name,
    PickChannel,
    RegionOf,
    Replace,
    ShapesOut,
    TableOut,
    tool,
)

FluoModel = Literal["2D_versatile_fluo", "2D_paper_dsb2018", "Custom model folder"]
RgbModel = Literal["2D_versatile_he", "Custom model folder"]

ModelFolder = Annotated[
    Folder | None,
    Group("Model"),
    EnabledWhen("model", "Custom model folder"),
    Description(
        "Folder of a StarDist 2D model you trained (contains config.json and weights_best.h5); "
        "used only when Model is 'Custom model folder'"
    ),
]
ProbThresh = Annotated[
    float | None,
    Min(0),
    Max(1),
    Group("Detection"),
    Description(
        "Probability threshold: higher gives fewer, more certain objects; unset uses the model's optimised value"
    ),
]
NmsThresh = Annotated[
    float | None,
    Min(0),
    Max(1),
    Group("Detection"),
    Description("Overlap threshold: higher allows more overlapping objects; unset uses the model's optimised value"),
]
LowPercentile = Annotated[
    float,
    Min(0),
    Max(100),
    Advanced(),
    Description("Lower intensity percentile for normalisation; 1 suits most images"),
]
HighPercentile = Annotated[
    float,
    Min(0),
    Max(100),
    Advanced(),
    Description("Upper intensity percentile for normalisation; 99.8 suits most images"),
]
SegmentOutputs = tuple[
    Annotated[LabelsOut, Name("stardist_labels"), ApplyTo("image"), Replace()],
    Annotated[ShapesOut, Name("stardist_outlines"), ApplyTo("image"), Replace()],
    Annotated[TableOut, Name("stardist_objects"), Replace()],
    Annotated[MessageOut, Name("summary")],
]


@tool("StarDist 2D: Segment nuclei (fluorescence)")
def segment_fluorescence(
    image: Annotated[Image, Axes("YX"), PickChannel(), Description("Single-channel image to segment")],
    model: Annotated[
        FluoModel,
        Group("Model"),
        Description("Pretrained model, or your own model folder; 2D_versatile_fluo suits most nuclei"),
    ] = "2D_versatile_fluo",
    model_folder: ModelFolder = None,
    prob_thresh: ProbThresh = None,
    nms_thresh: NmsThresh = None,
    region: Annotated[
        Labels | None,
        RegionOf("image"),
        Description("Segment only inside the selected region (Fiji ROI, napari shapes, QuPath annotation)"),
    ] = None,
    low_percentile: LowPercentile = 1.0,
    high_percentile: HighPercentile = 99.8,
) -> SegmentOutputs:
    """Segment star-convex objects in a single-channel image with StarDist 2D."""
    from stardist2d import bridge

    return bridge.segment(
        image,
        bridge.ModelChoice(model, model_folder),
        bridge.DetectionSettings(prob_thresh, nms_thresh, low_percentile, high_percentile),
        region=region,
    )


@tool("StarDist 2D: Segment nuclei (H&E / RGB)")
def segment_rgb(
    image: Annotated[Image, Axes("YXS"), Description("RGB image to segment, for example H&E")],
    model: Annotated[
        RgbModel,
        Group("Model"),
        Description("Pretrained H&E model, or your own 3-channel model folder"),
    ] = "2D_versatile_he",
    model_folder: ModelFolder = None,
    prob_thresh: ProbThresh = None,
    nms_thresh: NmsThresh = None,
    region: Annotated[
        Labels | None,
        RegionOf("image"),
        Description("Segment only inside the selected region (Fiji ROI, napari shapes, QuPath annotation)"),
    ] = None,
    low_percentile: LowPercentile = 1.0,
    high_percentile: HighPercentile = 99.8,
) -> SegmentOutputs:
    """Segment nuclei in an RGB (brightfield H&E) image with StarDist 2D."""
    from stardist2d import bridge

    return bridge.segment(
        image,
        bridge.ModelChoice(model, model_folder),
        bridge.DetectionSettings(prob_thresh, nms_thresh, low_percentile, high_percentile),
        region=region,
    )


@tool("StarDist 2D: Segment a folder")
def segment_folder(
    input_folder: Annotated[Folder, Description("Folder of 2D TIFF images to segment")],
    output_folder: Annotated[
        Folder,
        Description("Existing folder for the label images, ROI zips and counts (never the input folder)"),
    ],
    model: Annotated[
        Literal["2D_versatile_fluo", "2D_versatile_he", "2D_paper_dsb2018", "Custom model folder"],
        Group("Model"),
        Description("Pretrained model, or your own model folder"),
    ] = "2D_versatile_fluo",
    model_folder: ModelFolder = None,
    prob_thresh: ProbThresh = None,
    nms_thresh: NmsThresh = None,
    export_rois: Annotated[
        bool, Description("Also write an ImageJ ROI zip for every image (opens in the Fiji ROI Manager)")
    ] = True,
) -> tuple[Annotated[TableOut, Name("object_counts")], Annotated[MessageOut, Name("summary")]]:
    """Segment every image of a folder, as section 6.1 of the notebook does."""
    from stardist2d import bridge

    return bridge.segment_folder(
        input_folder,
        output_folder,
        bridge.ModelChoice(model, model_folder),
        bridge.DetectionSettings(prob_thresh, nms_thresh, 1.0, 99.8),
        export_rois=export_rois,
    )


@tool("StarDist 2D: Quality control")
def quality_control(
    image: Annotated[Image, Axes("YX"), PickChannel(), Description("Test image")],
    ground_truth: Annotated[Labels, Description("Ground-truth label image of the same field")],
    model: Annotated[
        FluoModel, Group("Model"), Description("Model to evaluate; choose 'Custom model folder' for your trained model")
    ] = "2D_versatile_fluo",
    model_folder: ModelFolder = None,
    iou_threshold: Annotated[
        float,
        Min(0.1),
        Max(1),
        Description("IoU above which a predicted object matches a true object; 0.5 is the usual value"),
    ] = 0.5,
) -> tuple[
    Annotated[LabelsOut, Name("prediction"), ApplyTo("image"), Replace()],
    Annotated[TableOut, Name("qc_metrics"), Replace()],
    Annotated[MessageOut, Name("summary")],
]:
    """Compare a model's prediction with ground truth (notebook section 5.2 metrics)."""
    from stardist2d import bridge

    return bridge.quality_control(image, ground_truth, bridge.ModelChoice(model, model_folder), iou_threshold)


@tool("StarDist 2D: Check this installation")
def check_installation() -> tuple[Annotated[MessageOut, Name("readout")], Annotated[TableOut, Name("details")]]:
    """Check memory, disk, GPU and TensorFlow in this app's environment."""
    from labconstrictor_tools import diagnostics

    from stardist2d import bridge

    checks = diagnostics.run_checks(benchmark=False, extra=[bridge.tensorflow_checks])
    return diagnostics.summary(checks), diagnostics.rows(checks)
