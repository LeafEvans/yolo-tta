import torch
import torchvision
import torch.nn.functional as F
from ultralytics.utils import ops
import torch.optim as optim
from pathlib import Path
from ultralytics.utils.loss import FeatureAlignmentLoss
from ultralytics.nn.modules import Adaptor
from ultralytics.utils import LOGGER


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


class TTAManager:
    def __init__(self, args, device, save_dir, model_path=None):
        self.args, self.device, self.save_dir = args, device, save_dir
        self.model_path = Path(model_path) if model_path else None
        self.stats = None
        self.criterion = None
        self.optimizer = None
        self.hook = None
        self.features = None

    def register_hook(self, module):
        self.hook = module.register_forward_hook(lambda m, inp, out: setattr(self, "features", out))

    def load_stats(self):
        candidates = [
            self.model_path.parent if self.model_path else None,
            self.save_dir.parent / "train",
            self.save_dir,
        ]
        for p in filter(None, candidates):
            f = p / "tta_stats.pt"
            if f.exists():
                LOGGER.info(f"Loaded TTA stats from {f}")
                self.stats = torch.load(f, map_location=self.device)
                return True
        LOGGER.warning("TTA stats not found, skipping TTA")
        return False

    def build(self, model, nc):
        fs = self.stats
        self.criterion = FeatureAlignmentLoss(
            nc=nc,
            feat_stats={
                "mu_img": fs["mu_img"],
                "inv_sigma_sq_img": fs["inv_sigma_sq_img"],
                "mu_obj_dict": fs["mu_obj_dict"],
                "inv_sigma_sq_obj_dict": fs["inv_sigma_sq_obj_dict"],
                "pi_obj": fs.get("pi_obj", None),
            },
            alpha=self.args.tta_alpha,
            alpha_ema_loss=self.args.tta_alpha_ema_loss,
            D_in_KL=fs.get("D_in_KL", 0.5),
            tau1=self.args.tta_tau1,
            tau2=self.args.tta_tau2,
        ).to(self.device)

        params = []
        for p in model.parameters():
            p.requires_grad_(False)
        if self.args.tta_update_mode == "adaptor":
            for m in model.modules():
                if isinstance(m, Adaptor):
                    for p in m.parameters():
                        p.requires_grad_(True)
                        params.append(p)
        elif self.args.tta_update_mode == "full":
            for p in model.parameters():
                p.requires_grad_(True)
                params.append(p)

        if not params:
            LOGGER.warning("No TTA parameters found, disabling TTA")
            return False

        self.optimizer = optim.Adam(params, lr=self.args.tta_lr)
        LOGGER.info(f"TTA will update {len(params)} parameters (mode={self.args.tta_update_mode})")
        return True

    def cleanup(self):
        if self.hook:
            self.hook.remove()
