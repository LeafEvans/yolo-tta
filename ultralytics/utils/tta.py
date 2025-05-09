import torch
import torchvision
import torch.nn.functional as F
from ultralytics.utils import ops


def extract_tta_features(
    im, preds_raw, conf, iou, classes, agnostic_nms, max_det, nc, rotated, tta_feature_layer, tta_conf_threshold
):
    preds, feats = preds_raw[:2]
    with torch.no_grad():
        dets_batch = ops.non_max_suppression(
            preds,
            conf,
            iou,
            classes,
            agnostic_nms,
            max_det=max_det,
            nc=nc,
            rotated=rotated,
        )
    idx = int(tta_feature_layer) if isinstance(tta_feature_layer, int) else 0
    fmap = feats[idx]
    B, C, H, W = fmap.shape
    F_img = F.adaptive_avg_pool2d(fmap, (1, 1)).flatten(1)
    device = fmap.device
    _, _, H_in, W_in = im.shape
    scale = fmap.new_tensor([W / W_in, H / H_in, W / W_in, H / H_in])
    rois_list, cls_list = [], []
    for i, det in enumerate(dets_batch):
        if det is None or det.numel() == 0:
            continue
        det = det[det[:, 4] > tta_conf_threshold]
        if det.numel() == 0:
            continue
        coords = det[:, :4] * scale
        b_idx = torch.full((coords.size(0), 1), float(i), device=device)
        rois_list.append(torch.cat([b_idx, coords], dim=1))
        cls_list.append(det[:, 5].long())
    if rois_list:
        rois = torch.cat(rois_list, dim=0)
        cls_flat = torch.cat(cls_list, dim=0)
        obj_feats = torchvision.ops.roi_align(fmap, rois, output_size=(1, 1), spatial_scale=1.0, aligned=True).view(
            -1, C
        )
        F_obj = {int(c): obj_feats[cls_flat == c] for c in torch.unique(cls_flat)}
    else:
        F_obj = {}
    return F_img, F_obj
