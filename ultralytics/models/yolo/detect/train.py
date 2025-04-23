# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from collections import defaultdict
from copy import copy
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first


class DetectionTrainer(BaseTrainer):
    """
    A class extending the BaseTrainer class for training based on a detection model.

    This trainer specializes in object detection tasks, handling the specific requirements for training YOLO models
    for object detection.

    Attributes:
        model (DetectionModel): The YOLO detection model being trained.
        data (dict): Dictionary containing dataset information including class names and number of classes.
        loss_names (Tuple[str]): Names of the loss components used in training (box_loss, cls_loss, dfl_loss).

    Methods:
        build_dataset: Build YOLO dataset for training or validation.
        get_dataloader: Construct and return dataloader for the specified mode.
        preprocess_batch: Preprocess a batch of images by scaling and converting to float.
        set_model_attributes: Set model attributes based on dataset information.
        get_model: Return a YOLO detection model.
        get_validator: Return a validator for model evaluation.
        label_loss_items: Return a loss dictionary with labeled training loss items.
        progress_string: Return a formatted string of training progress.
        plot_training_samples: Plot training samples with their annotations.
        plot_metrics: Plot metrics from a CSV file.
        plot_training_labels: Create a labeled training plot of the YOLO model.
        auto_batch: Calculate optimal batch size based on model memory requirements.

    Examples:
        >>> from ultralytics.models.yolo.detect import DetectionTrainer
        >>> args = dict(model="yolo11n.pt", data="coco8.yaml", epochs=3)
        >>> trainer = DetectionTrainer(overrides=args)
        >>> trainer.train()
    """

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Dataset for training or validation.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`.

        Returns:
            (Dataset): YOLO dataset object configured for the specified mode.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """
        Construct and return dataloader for the specified mode.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Number of images per batch.
            rank (int): Process rank for distributed training.
            mode (str): 'train' for training dataloader, 'val' for validation dataloader.

        Returns:
            (DataLoader): PyTorch dataloader object.
        """
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader

    def preprocess_batch(self, batch):
        """
        Preprocess a batch of images by scaling and converting to float.

        Args:
            batch (dict): Dictionary containing batch data with 'img' tensor.

        Returns:
            (dict): Preprocessed batch with normalized images.
        """
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        if self.args.multi_scale:
            imgs = batch["img"]
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def set_model_attributes(self):
        """Set model attributes based on dataset information."""
        # Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.names = self.data["names"]  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Return a YOLO detection model.

        Args:
            cfg (str, optional): Path to model configuration file.
            weights (str, optional): Path to model weights.
            verbose (bool): Whether to display model information.

        Returns:
            (DetectionModel): YOLO detection model.
        """
        model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Return a loss dict with labeled training loss items tensor.

        Args:
            loss_items (List[float], optional): List of loss values.
            prefix (str): Prefix for keys in the returned dictionary.

        Returns:
            (Dict | List): Dictionary of labeled loss items if loss_items is provided, otherwise list of keys.
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Return a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch, ni):
        """
        Plot training samples with their annotations.

        Args:
            batch (dict): Dictionary containing batch data.
            ni (int): Number of iterations.
        """
        plot_images(
            images=batch["img"],
            batch_idx=batch["batch_idx"],
            cls=batch["cls"].squeeze(-1),
            bboxes=batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        """Plot metrics from a CSV file."""
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def auto_batch(self):
        """
        Get optimal batch size by calculating memory occupation of model.

        Returns:
            (int): Optimal batch size.
        """
        train_dataset = self.build_dataset(self.trainset, mode="train", batch=16)
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4  # 4 for mosaic augmentation
        return super().auto_batch(max_num_obj)

    def _extract_features_for_tta(self, model, batch, feature_layer_idx=-1):
        """
        Extract features for TTA statistics calculation from the detection model.
        Handles large numbers of RoIs by batching the roi_align call for robustness.

        Args:
            model (nn.Module): The model (potentially EMA version) to extract features from.
            batch (dict): A batch of data containing images and ground truth annotations.
            feature_layer_idx (int): Index of the layer to extract features from.
                                     For DetectionModel, -1 typically uses the first neck output.

        Returns:
            (tuple): A tuple containing:
                - F_img (torch.Tensor | None): Image-level features [N, C]. None if extraction fails.
                - F_obj_dict_gt (dict[int, torch.Tensor] | None): Dictionary mapping class index to
                  concatenated object-level features [N_obj_cls, C]. None if extraction fails or
                  roi_align is unavailable.
        """
        model_eval = de_parallel(model).eval()
        feature_map = None
        F_img = None
        F_obj_dict_gt = None
        stride = None

        try:
            img_tensor = batch["img"].to(self.device, non_blocking=True)
            _, feats = model_eval(img_tensor)

            if not isinstance(feats, (list, tuple)) or not feats:
                LOGGER.error("Model output 'feats' is not a valid list/tuple or is empty.")
                return None, None

            valid_indices = list(range(len(feats))) + [-1]
            if feature_layer_idx not in valid_indices:
                LOGGER.error(f"Feature layer index {feature_layer_idx} is out of bounds for {len(feats)} feature maps.")
                return None, None

            actual_idx = 0 if feature_layer_idx == -1 else feature_layer_idx
            feature_map = feats[actual_idx]
            model_strides = getattr(model_eval, "stride", torch.tensor([8.0, 16.0, 32.0]))
            if actual_idx < len(model_strides):
                stride = model_strides[actual_idx].item()
                LOGGER.debug(f"Using feature map at index {actual_idx} with stride {stride:.1f} for TTA.")
            else:
                LOGGER.warning(f"Could not determine stride for feature map index {actual_idx}.")

        except Exception as e:
            LOGGER.error(f"Error during model forward pass for feature extraction: {e}", exc_info=True)
            return None, None

        if feature_map is not None:
            try:
                F_img = F.adaptive_avg_pool2d(feature_map, (1, 1)).flatten(start_dim=1)
            except Exception as e:
                LOGGER.error(f"Error during image-level feature pooling: {e}", exc_info=True)
                F_img = None
        else:
            LOGGER.warning("Feature map is None, cannot extract image-level features.")

        can_extract_obj = (
            roi_align is not None
            and feature_map is not None
            and all(k in batch for k in ["bboxes", "cls", "batch_idx"])
        )

        if not can_extract_obj:
            if roi_align is None:
                LOGGER.warning("torchvision.ops.roi_align not found. Object-level TTA features cannot be extracted.")
            elif feature_map is None:
                LOGGER.warning("Feature map is None, cannot extract object-level features.")
            else:
                LOGGER.warning("Ground truth 'bboxes', 'cls', or 'batch_idx' not found. Skipping TTA object features.")
        else:
            gt_bboxes = batch["bboxes"].to(feature_map.device)
            gt_classes = batch["cls"].squeeze(-1).int().to(feature_map.device)
            gt_batch_idx = batch["batch_idx"].to(feature_map.device)
            num_gt = gt_bboxes.shape[0]

            if num_gt == 0:
                LOGGER.debug("No ground truth objects in this batch for TTA object feature extraction.")
            else:
                fm_h, fm_w = feature_map.shape[2:]
                scaling_tensor = feature_map.new_tensor([[fm_w, fm_h, fm_w, fm_h]])

                boxes_fm = gt_bboxes * scaling_tensor
                boxes_for_roi = torch.cat([gt_batch_idx.unsqueeze(-1).float(), boxes_fm], dim=1)

                MAX_ROIS_PER_CALL = 500
                all_roi_features_list = []
                all_gt_classes_list = []

                for start_idx in range(0, num_gt, MAX_ROIS_PER_CALL):
                    end_idx = min(start_idx + MAX_ROIS_PER_CALL, num_gt)
                    chunk_boxes_for_roi_orig = boxes_for_roi[start_idx:end_idx]
                    chunk_gt_classes = gt_classes[start_idx:end_idx]
                    num_in_chunk = chunk_boxes_for_roi_orig.shape[0]

                    if num_in_chunk == 0:
                        continue

                    scaled_x1 = chunk_boxes_for_roi_orig[:, 1]
                    scaled_y1 = chunk_boxes_for_roi_orig[:, 2]
                    scaled_x2 = chunk_boxes_for_roi_orig[:, 3]
                    scaled_y2 = chunk_boxes_for_roi_orig[:, 4]
                    pre_clamp_valid_mask = (scaled_x2 - scaled_x1 >= 0.5) & (scaled_y2 - scaled_y1 >= 0.5)

                    if not pre_clamp_valid_mask.any():
                        continue

                    boxes_roi_pre_filtered = chunk_boxes_for_roi_orig[pre_clamp_valid_mask]
                    gt_classes_pre_filtered = chunk_gt_classes[pre_clamp_valid_mask]

                    clamped_x1 = torch.clamp(boxes_roi_pre_filtered[:, 1], 0, fm_w - 1)
                    clamped_y1 = torch.clamp(boxes_roi_pre_filtered[:, 2], 0, fm_h - 1)
                    clamped_x2 = torch.clamp(boxes_roi_pre_filtered[:, 3], 0, fm_w - 1)
                    clamped_y2 = torch.clamp(boxes_roi_pre_filtered[:, 4], 0, fm_h - 1)

                    post_clamp_valid_mask = (clamped_x2 > clamped_x1) & (clamped_y2 > clamped_y1)

                    if not post_clamp_valid_mask.any():
                        continue

                    final_valid_batch_indices = boxes_roi_pre_filtered[post_clamp_valid_mask, 0]
                    final_valid_gt_classes = gt_classes_pre_filtered[post_clamp_valid_mask]
                    final_valid_boxes_roi = torch.stack(
                        [
                            final_valid_batch_indices,
                            clamped_x1[post_clamp_valid_mask],
                            clamped_y1[post_clamp_valid_mask],
                            clamped_x2[post_clamp_valid_mask],
                            clamped_y2[post_clamp_valid_mask],
                        ],
                        dim=1,
                    )

                    try:
                        chunk_roi_features = roi_align(
                            feature_map,
                            final_valid_boxes_roi,
                            output_size=(1, 1),
                            spatial_scale=1.0,
                            sampling_ratio=-1,
                            aligned=True,
                        ).flatten(1)

                        all_roi_features_list.append(chunk_roi_features)
                        all_gt_classes_list.append(final_valid_gt_classes)
                    except Exception as e:
                        LOGGER.warning(f"Error during roi_align in chunk {start_idx}-{end_idx}: {e}", exc_info=True)

                if all_roi_features_list:
                    roi_features_all = torch.cat(all_roi_features_list, dim=0)
                    gt_classes_all = torch.cat(all_gt_classes_list, dim=0)

                    final_dict = defaultdict(list)
                    for i in range(roi_features_all.shape[0]):
                        cls_idx = gt_classes_all[i].item()
                        final_dict[cls_idx].append(roi_features_all[i])

                    F_obj_dict_gt = {cls_idx: torch.stack(feats_list) for cls_idx, feats_list in final_dict.items()}
                else:
                    F_obj_dict_gt = None

        img_status = "extracted" if F_img is not None else "failed/skipped"
        obj_status = "extracted" if F_obj_dict_gt else "failed/skipped"
        LOGGER.debug(f"TTA Feature Extraction: Image-level {img_status}, Object-level {obj_status}.")

        return F_img, F_obj_dict_gt
