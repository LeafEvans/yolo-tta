# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ops


class DetectionPredictor(BasePredictor):
    """
    Detection predictor with optional Test-Time Adaptation (TTA).
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)

    def _extract_features_for_tta(self, im, preds_raw):
        from ultralytics.utils.tta import extract_tta_features

        return extract_tta_features(
            im,
            preds_raw,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            self.args.max_det,
            len(self.model.names),
            (self.args.task == "obb"),
            self.args.tta_feature_layer,
            self.args.tta_conf_threshold,
        )

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        # standard NMS
        preds_input = preds[0] if isinstance(preds, (list, tuple)) else preds
        preds = ops.non_max_suppression(
            preds_input,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            rotated=self.args.task == "obb",
        )

        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        return self.construct_results(preds, img, orig_imgs)

    def construct_results(self, preds, img, orig_imgs):
        return [
            self.construct_result(pred, img, orig_img, path)
            for pred, orig_img, path in zip(preds, orig_imgs, self.batch[0])
        ]

    def construct_result(self, pred, img, orig_img, img_path):
        pred = pred.cpu()
        if self.args.task == "obb":
            pred[:, :5] = ops.scale_coords_rotated(img.shape[2:], pred[:, :5], orig_img.shape)
        else:
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)

        box_data = pred[:, : (5 if self.args.task == "obb" else 4)]
        conf_cls = pred[:, -2:]
        boxes = torch.cat((box_data, conf_cls), 1)
        return Results(orig_img, path=img_path, names=self.model.names, boxes=boxes)
