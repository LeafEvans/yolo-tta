# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import torch.nn.functional as F
import torchvision

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
        # Split preds_tensor and feats
        if isinstance(preds_raw, (list, tuple)) and len(preds_raw) == 2:
            preds_tensor, feats = preds_raw
        else:
            out = self.model.model(im)
            if not (isinstance(out, (list, tuple)) and len(out) == 2):
                return None, {}
            preds_tensor, feats = out

        # Apply NMS to original preds_tensor
        with torch.no_grad():
            dets = ops.non_max_suppression(
                preds_tensor,
                self.args.conf,
                self.args.iou,
                classes=self.args.classes,
                agnostic=self.args.agnostic_nms,
                max_det=self.args.max_det,
                nc=len(self.model.names),
                rotated=self.args.task == "obb",
            )[0]
        if dets is None or dets.numel() == 0:
            F_img = F.adaptive_avg_pool2d(feats[0], (1, 1)).view(1, -1)
            return F_img, {}

        # Construct boxes object
        from types import SimpleNamespace

        B = feats[0].shape[0] if isinstance(feats, (list, tuple)) else 1
        xy = dets[:, :4].view(1, -1, 4)
        cls = dets[:, 5].long().view(1, -1)
        conf = dets[:, 4].view(1, -1)
        boxes = SimpleNamespace(xyxy=xy, cls=cls, conf=conf)

        idx = int(self.args.tta_feature_layer)
        actual_idx = 0 if idx == -1 else idx if 0 <= idx < len(feats) else None
        if actual_idx is None:
            return None, {}

        fmap = feats[actual_idx]
        B, C, H, W = fmap.shape
        F_img = F.adaptive_avg_pool2d(fmap, (1, 1)).view(B, C)

        device, dtype = fmap.device, fmap.dtype
        _, _, H_in, W_in = im.shape
        scale = torch.as_tensor([W / W_in, H / H_in, W / W_in, H / H_in], device=device, dtype=dtype)
        xyxy = boxes.xyxy.to(device) if boxes.xyxy.device != device else boxes.xyxy
        cls_t = boxes.cls.long().to(device) if boxes.cls.device != device else boxes.cls.long()

        confs = getattr(boxes, "conf", None)
        if confs is not None:
            confs = confs.to(device) if confs.device != device else confs
            mask = confs > self.args.tta_conf_threshold
        else:
            mask = torch.ones_like(cls_t, dtype=torch.bool, device=device)

        rois, cls_idx = [], []
        for b in range(B):
            m = mask[b]
            if not m.any():
                continue
            coords = xyxy[b][m] * scale
            bi = torch.full((coords.size(0), 1), b, device=device, dtype=coords.dtype)
            rois.append(torch.cat([bi, coords], dim=1))
            cls_idx.append(cls_t[b][m])

        if not rois:
            return F_img, {}

        rois = torch.cat(rois, dim=0)
        cls_idx = torch.cat(cls_idx, dim=0)

        obj_feats = torchvision.ops.roi_align(fmap, rois, output_size=(1, 1), spatial_scale=1.0, aligned=True).view(
            -1, C
        )

        F_obj = {}
        for c in cls_idx.unique():
            mask_c = cls_idx == c
            if mask_c.any():
                F_obj[int(c.item())] = obj_feats[mask_c]

        return F_img, F_obj

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
