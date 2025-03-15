# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path

from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, OBBModel, PoseModel, SegmentationModel, WorldModel
from ultralytics.utils import ROOT, LOGGER, yaml_load


class YOLO(Model):
    """YOLO (You Only Look Once) object detection model."""

    def __init__(self, model="yolo11n.pt", task=None, verbose=False):
        """Initialize YOLO model, switching to YOLOWorld if model filename contains '-world'."""
        path = Path(model)
        if "-world" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:  # if YOLOWorld PyTorch model
            new_instance = YOLOWorld(path, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        else:
            # Continue with default YOLO initialization
            super().__init__(model=model, task=task, verbose=verbose)

        # Initialize TTA settings
        self._tta_enabled = False
        self._tta_settings = {}

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "classify": {
                "model": ClassificationModel,
                "trainer": yolo.classify.ClassificationTrainer,
                "validator": yolo.classify.ClassificationValidator,
                "predictor": yolo.classify.ClassificationPredictor,
            },
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
            "segment": {
                "model": SegmentationModel,
                "trainer": yolo.segment.SegmentationTrainer,
                "validator": yolo.segment.SegmentationValidator,
                "predictor": yolo.segment.SegmentationPredictor,
            },
            "pose": {
                "model": PoseModel,
                "trainer": yolo.pose.PoseTrainer,
                "validator": yolo.pose.PoseValidator,
                "predictor": yolo.pose.PosePredictor,
            },
            "obb": {
                "model": OBBModel,
                "trainer": yolo.obb.OBBTrainer,
                "validator": yolo.obb.OBBValidator,
                "predictor": yolo.obb.OBBPredictor,
            },
        }

    def enable_tta(self):
        """Enable Test-Time Adaptation for the model."""
        self._tta_enabled = True
        if hasattr(self, "predictor") and self.predictor is not None:
            self.predictor.args.tta = True
        return self

    def disable_tta(self):
        """Disable Test-Time Adaptation for the model."""
        self._tta_enabled = False
        if hasattr(self, "predictor") and self.predictor is not None:
            self.predictor.args.tta = False
        return self

    def configure_tta(self, **kwargs):
        """
        Configure Test-Time Adaptation parameters.

        Args:
            **kwargs: Keyword arguments for TTA configuration.
                - alpha (float): EMA update rate (default: 0.01)
                - tau1 (float): Primary threshold (default: 1.1)
                - tau2 (float): Secondary threshold (default: 1.05)
                - momentum (float): EMA momentum (default: 0.99)
                - reduction_ratio (int): Reduction ratio for adaptors (default: 32)
                - tta_lr (float): Learning rate for TTA updates (default: 0.001)
                - feature_layer (int): Index of the layer to extract features from (default: -3)
        """
        valid_params = {
            "alpha": "tta_alpha",
            "tau1": "tta_tau1",
            "tau2": "tta_tau2",
            "momentum": "tta_momentum",
            "reduction_ratio": "tta_reduction_ratio",
            "tta_lr": "tta_lr",
            "feature_layer": "tta_feature_layer",
        }

        for k, v in kwargs.items():
            if k in valid_params:
                arg_name = valid_params[k]
                self._tta_settings[arg_name] = v
                # Update predictor if already created
                if hasattr(self, "predictor") and self.predictor is not None:
                    setattr(self.predictor.args, arg_name, v)
            else:
                LOGGER.warning(f"Unknown TTA parameter: {k}")

        return self

    def _ensure_predictor(self):
        """Ensures predictor is created with the right TTA configuration."""
        # Call the parent method first
        super()._ensure_predictor()

        # Apply TTA settings
        if self.predictor is not None:
            if self._tta_enabled:
                self.predictor.args.tta = True

            # Apply all configured TTA settings
            for k, v in self._tta_settings.items():
                setattr(self.predictor.args, k, v)


class YOLOWorld(Model):
    """YOLO-World object detection model."""

    def __init__(self, model="yolov8s-world.pt", verbose=False) -> None:
        """
        Initialize YOLOv8-World model with a pre-trained model file.

        Loads a YOLOv8-World model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """
        super().__init__(model=model, task="detect", verbose=verbose)

        # Assign default COCO class names when there are no custom names
        if not hasattr(self.model, "names"):
            self.model.names = yaml_load(ROOT / "cfg/datasets/coco8.yaml").get("names")

        # Initialize TTA settings
        self._tta_enabled = False
        self._tta_settings = {}

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": WorldModel,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
                "trainer": yolo.world.WorldTrainer,
            }
        }

    def set_classes(self, classes):
        """
        Set classes.

        Args:
            classes (List(str)): A list of categories i.e. ["person"].
        """
        self.model.set_classes(classes)
        # Remove background if it's given
        background = " "
        if background in classes:
            classes.remove(background)
        self.model.names = classes

        # Reset method class names
        # self.predictor = None  # reset predictor otherwise old names remain
        if self.predictor:
            self.predictor.model.names = classes

    # 继承TTA方法
    enable_tta = YOLO.enable_tta
    disable_tta = YOLO.disable_tta
    configure_tta = YOLO.configure_tta
    _ensure_predictor = YOLO._ensure_predictor
