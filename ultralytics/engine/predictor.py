# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Run prediction on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ yolo mode=predict model=yolo11n.pt source=0                               # webcam
                                                img.jpg                         # image
                                                vid.mp4                         # video
                                                screen                          # screenshot
                                                path/                           # directory
                                                list.txt                        # list of images
                                                list.streams                    # list of streams
                                                'path/*.jpg'                    # glob
                                                'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP, TCP stream

Usage - formats:
    $ yolo mode=predict model=yolo11n.pt                 # PyTorch
                              yolo11n.torchscript        # TorchScript
                              yolo11n.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                              yolo11n_openvino_model     # OpenVINO
                              yolo11n.engine             # TensorRT
                              yolo11n.mlpackage          # CoreML (macOS-only)
                              yolo11n_saved_model        # TensorFlow SavedModel
                              yolo11n.pb                 # TensorFlow GraphDef
                              yolo11n.tflite             # TensorFlow Lite
                              yolo11n_edgetpu.tflite     # TensorFlow Edge TPU
                              yolo11n_paddle_model       # PaddlePaddle
                              yolo11n.mnn                # MNN
                              yolo11n_ncnn_model         # NCNN
                              yolo11n_imx_model          # Sony IMX
                              yolo11n_rknn_model         # Rockchip RKNN
"""

import platform
import re
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import optim

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data import load_inference_source
from ultralytics.data.augment import LetterBox, classify_transforms
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.modules import Adaptor
from ultralytics.utils import DEFAULT_CFG, LOGGER, MACOS, WINDOWS, callbacks, colorstr, ops
from ultralytics.utils.checks import check_imgsz, check_imshow
from ultralytics.utils.files import increment_path
from ultralytics.utils.torch_utils import select_device
from ultralytics.utils.loss import FeatureAlignmentLoss

STREAM_WARNING = """
WARNING ⚠️ inference results will accumulate in RAM unless `stream=True` is passed, causing potential out-of-memory
errors for large sources or long-running streams and videos. See https://docs.ultralytics.com/modes/predict/ for help.

Example:
    results = model(source=..., stream=True)  # generator of Results objects
    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs
        probs = r.probs  # Class probabilities for classification outputs
"""


class BasePredictor:
    """
    A base class for creating predictors.

    This class provides the foundation for prediction functionality, handling model setup, inference,
    and result processing across various input sources.

    Attributes:
        args (SimpleNamespace): Configuration for the predictor.
        save_dir (Path): Directory to save results.
        done_warmup (bool): Whether the predictor has finished setup.
        model (torch.nn.Module): Model used for prediction.
        data (dict): Data configuration.
        device (torch.device): Device used for prediction.
        dataset (Dataset): Dataset used for prediction.
        vid_writer (dict): Dictionary of {save_path: video_writer} for saving video output.
        plotted_img (numpy.ndarray): Last plotted image.
        source_type (SimpleNamespace): Type of input source.
        seen (int): Number of images processed.
        windows (list): List of window names for visualization.
        batch (tuple): Current batch data.
        results (list): Current batch results.
        transforms (callable): Image transforms for classification.
        callbacks (dict): Callback functions for different events.
        txt_path (Path): Path to save text results.
        _lock (threading.Lock): Lock for thread-safe inference.

    Methods:
        preprocess: Prepare input image before inference.
        inference: Run inference on a given image.
        postprocess: Process raw predictions into structured results.
        predict_cli: Run prediction for command line interface.
        setup_source: Set up input source and inference mode.
        stream_inference: Stream inference on input source.
        setup_model: Initialize and configure the model.
        write_results: Write inference results to files.
        save_predicted_images: Save prediction visualizations.
        show: Display results in a window.
        run_callbacks: Execute registered callbacks for an event.
        add_callback: Register a new callback function.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize the BasePredictor class.

        Args:
            cfg (str | dict): Path to a configuration file or a configuration dictionary.
            overrides (dict | None): Configuration overrides.
            _callbacks (dict | None): Dictionary of callback functions.
        """
        self.args = get_cfg(cfg, overrides)
        self.save_dir = get_save_dir(self.args)
        if self.args.conf is None:
            self.args.conf = 0.25  # default conf=0.25
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        # TTA specific args
        self.args.tta = getattr(self.args, "tta", False)
        self.args.tta_feature_layer = getattr(self.args, "tta_feature_layer", -1)
        self.args.tta_conf_threshold = getattr(self.args, "tta_conf_threshold", 0.5)
        self.args.tta_lr = getattr(self.args, "tta_lr", 0.001)
        self.args.tta_alpha = getattr(self.args, "tta_alpha", 0.01)
        self.args.tta_alpha_ema_loss = getattr(self.args, "tta_alpha_ema_loss", 0.01)
        self.args.tta_tau1 = getattr(self.args, "tta_tau1", 1.1)
        self.args.tta_tau2 = getattr(self.args, "tta_tau2", 1.05)
        self.args.tta_update_mode = getattr(self.args, "tta_update_mode", "adaptor").lower()
        self.args.tta_loss_mode = getattr(self.args, "tta_loss_mode", "weighted_obj").lower()
        self.args.tta_update_strategy = getattr(self.args, "tta_update_strategy", "conditional").lower()

        # Usable if setup is done
        self.model = None
        self.data = self.args.data  # data_dict
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_writer = {}  # dict of {save_path: video_writer, ...}
        self.plotted_img = None
        self.source_type = None
        self.seen = 0
        self.windows = []
        self.batch = None
        self.results = None
        self.transforms = None
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        self.txt_path = None
        self._lock = threading.Lock()  # for automatic thread-safe inference

        # TTA state variables
        self.tta_enabled = False
        self.tta_criterion = None
        self.tta_optimizer = None
        self.tta_stats = None
        self.done_tta_warmup = False

        callbacks.add_integration_callbacks(self)

    def preprocess(self, im):
        """
        Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): Images of shape (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.
        """
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        im = im.to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        if not_tensor:
            im /= 255  # 0 - 255 to 0.0 - 1.0
        return im

    def inference(self, im, *args, **kwargs):
        """Run inference on a given image using the specified model and arguments."""
        visualize = (
            increment_path(self.save_dir / Path(self.batch[0][0]).stem, mkdir=True)
            if self.args.visualize and (not self.source_type.tensor)
            else False
        )
        embed_arg = None if self.tta_enabled else self.args.embed
        return self.model(im, augment=self.args.augment, visualize=visualize, embed=embed_arg, *args, **kwargs)

    def pre_transform(self, im):
        """
        Pre-transform input image before inference.

        Args:
            im (List[np.ndarray]): Images of shape (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

        Returns:
            (List[np.ndarray]): A list of transformed images.
        """
        same_shapes = len({x.shape for x in im}) == 1
        letterbox = LetterBox(
            self.imgsz,
            auto=same_shapes and (self.model.pt or (getattr(self.model, "dynamic", False) and not self.model.imx)),
            stride=self.model.stride,
        )
        return [letterbox(image=x) for x in im]

    def postprocess(self, preds, img, orig_imgs):
        """Post-process predictions for an image and return them."""
        return preds

    def __call__(self, source=None, model=None, stream=False, *args, **kwargs):
        """
        Perform inference on an image or stream.

        Args:
            source (str | Path | List[str] | List[Path] | List[np.ndarray] | np.ndarray | torch.Tensor | None):
                Source for inference.
            model (str | Path | torch.nn.Module | None): Model for inference.
            stream (bool): Whether to stream the inference results. If True, returns a generator.
            *args (Any): Additional arguments for the inference method.
            **kwargs (Any): Additional keyword arguments for the inference method.

        Returns:
            (List[ultralytics.engine.results.Results] | generator): Results objects or generator of Results objects.
        """
        self.stream = stream
        if stream:
            return self.stream_inference(source, model, *args, **kwargs)
        else:
            return list(self.stream_inference(source, model, *args, **kwargs))  # merge list of Result into one

    def predict_cli(self, source=None, model=None):
        """
        Method used for Command Line Interface (CLI) prediction.

        This function is designed to run predictions using the CLI. It sets up the source and model, then processes
        the inputs in a streaming manner. This method ensures that no outputs accumulate in memory by consuming the
        generator without storing results.

        Args:
            source (str | Path | List[str] | List[Path] | List[np.ndarray] | np.ndarray | torch.Tensor | None):
                Source for inference.
            model (str | Path | torch.nn.Module | None): Model for inference.

        Note:
            Do not modify this function or remove the generator. The generator ensures that no outputs are
            accumulated in memory, which is critical for preventing memory issues during long-running predictions.
        """
        gen = self.stream_inference(source, model)
        for _ in gen:  # sourcery skip: remove-empty-nested-block, noqa
            pass

    def setup_source(self, source):
        """
        Set up source and inference mode.

        Args:
            source (str | Path | List[str] | List[Path] | List[np.ndarray] | np.ndarray | torch.Tensor):
                Source for inference.
        """
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)  # check image size
        self.transforms = (
            getattr(
                self.model.model,
                "transforms",
                classify_transforms(self.imgsz[0], crop_fraction=self.args.crop_fraction),
            )
            if self.args.task == "classify"
            else None
        )
        self.dataset = load_inference_source(
            source=source,
            batch=self.args.batch,
            vid_stride=self.args.vid_stride,
            buffer=self.args.stream_buffer,
        )
        self.source_type = self.dataset.source_type
        if not getattr(self, "stream", True) and (
            self.source_type.stream
            or self.source_type.screenshot
            or len(self.dataset) > 1000  # many images
            or any(getattr(self.dataset, "video_flag", [False]))
        ):  # videos
            LOGGER.warning(STREAM_WARNING)
        self.vid_writer = {}

    def stream_inference(self, source=None, model=None, *args, **kwargs):
        """
        Stream real-time inference on camera feed and save results to file.
        Includes TTA logic if enabled.

        Args:
            source (str | Path | List[str] | List[Path] | List[np.ndarray] | np.ndarray | torch.Tensor | None):
                Source for inference.
            model (str | Path | torch.nn.Module | None): Model for inference.
            *args (Any): Additional arguments for the inference method.
            **kwargs (Any): Additional keyword arguments for the inference method.

        Yields:
            (ultralytics.engine.results.Results): Results objects.
        """
        if self.args.verbose:
            LOGGER.info("")

        # Setup model
        if not self.model:
            self.setup_model(model)

        with self._lock:  # for thread-safe inference
            # Setup source every time predict is called
            self.setup_source(source if source is not None else self.args.source)

            # Check if save_dir/ label file exists
            if self.args.save or self.args.save_txt:
                (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

            # Warmup model
            if not self.done_warmup:
                self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
                self.done_warmup = True

            self.seen, self.windows, self.batch = 0, [], None
            profilers = (
                ops.Profile(device=self.device),  # 0: Preprocess
                ops.Profile(device=self.device),  # 1: Inference
                ops.Profile(device=self.device),  # 2: Postprocess
                ops.Profile(device=self.device),  # 3: TTA Feature Extraction
                ops.Profile(device=self.device),  # 4: TTA Loss & Update
            )
            for p in profilers:
                p.dt = 0.0
            self.run_callbacks("on_predict_start")
            for self.batch in self.dataset:
                self.run_callbacks("on_predict_batch_start")
                paths, im0s, s = self.batch

                with profilers[0]:
                    im = self.preprocess(im0s)

                with torch.set_grad_enabled(self.tta_enabled):
                    with profilers[1]:
                        raw = self.inference(im, *args, **kwargs)

                if self.tta_enabled:
                    with profilers[3]:
                        F_img, F_obj = self._extract_features_for_tta(im, raw)
                    if F_img is not None:
                        loss_val, _, update_flag = self.tta_criterion(F_img, F_obj, loss_mode=self.args.tta_loss_mode)
                        if loss_val > 0:
                            self.tta_optimizer.zero_grad()
                            with profilers[4]:
                                loss_val.backward()
                                perform_step = self.args.tta_update_strategy == "always" or (
                                    self.args.tta_update_strategy == "conditional" and update_flag
                                )
                                if perform_step:
                                    self.tta_optimizer.step()

                with profilers[2]:
                    with torch.no_grad():
                        preds = raw
                        self.results = self.postprocess(preds, im, im0s)

                self.run_callbacks("on_predict_postprocess_end")

                n = len(im0s)
                for i in range(n):
                    self.seen += 1
                    speed_metrics = {
                        "preprocess": profilers[0].dt * 1e3 / n,
                        "inference": profilers[1].dt * 1e3 / n,
                        "postprocess": profilers[2].dt * 1e3 / n,
                    }
                    if self.tta_enabled:
                        speed_metrics["tta_features"] = profilers[3].dt * 1e3 / n
                        speed_metrics["tta_update"] = profilers[4].dt * 1e3 / n

                    self.results[i].speed = speed_metrics
                    if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                        s[i] += self.write_results(i, Path(paths[i]), im, s)

                if self.args.verbose:
                    LOGGER.info("\n".join(s))

                self.run_callbacks("on_predict_batch_end")
                yield from self.results

        for v in self.vid_writer.values():
            if isinstance(v, cv2.VideoWriter):
                v.release()

        if self.args.verbose and self.seen:
            t_pre = profilers[0].t / self.seen * 1e3
            t_inf = profilers[1].t / self.seen * 1e3
            t_post = profilers[2].t / self.seen * 1e3
            speed_str = f"Speed: {t_pre:.1f}ms preprocess, {t_inf:.1f}ms inference, {t_post:.1f}ms postprocess"
            if self.tta_enabled:
                t_tta_feat = profilers[3].t / self.seen * 1e3
                t_tta_upd = profilers[4].t / self.seen * 1e3  # Includes backward + potential step
                speed_str += f", {t_tta_feat:.1f}ms TTA feats, {t_tta_upd:.1f}ms TTA update"
            speed_str += f" per image at shape {(min(self.args.batch, self.seen), 3, *im.shape[2:])}"
            LOGGER.info(speed_str)

        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob("labels/*.txt")))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ""
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
        self.run_callbacks("on_predict_end")

    def _extract_features_for_tta(self, im, preds_raw):
        """
        Placeholder for extracting features for TTA. Must be implemented by subclasses.

        Args:
            im (torch.Tensor): Preprocessed image tensor.
            preds_raw (Any): Raw predictions from the model's inference step, potentially before NMS.

        Returns:
            tuple(torch.Tensor | None, dict | None):
                - F_img: Image-level features.
                - F_obj_dict: Object-level features dictionary {cls_idx: features}.
        """
        return None, None

    def setup_model(self, model, verbose=True):
        """
        Initialize YOLO model with given parameters and set it to evaluation mode.

        Args:
            model (str | Path | torch.nn.Module | None): Model to load or use.
            verbose (bool): Whether to print verbose output.
        """
        # Standard model setup
        self.model = AutoBackend(
            weights=model or self.args.model,
            device=select_device(self.args.device, verbose=verbose),
            dnn=self.args.dnn,
            data=self.args.data,
            fp16=self.args.half,
            batch=self.args.batch,
            fuse=True,
            verbose=verbose,
        )
        self.device = self.model.device
        self.args.half = self.model.fp16
        self.model.eval()

        if self.args.tta:
            self.tta_features = None

            backend = getattr(self.model, "backend", "pytorch")
            module = self.model.model if backend == "pytorch" and hasattr(self.model, "model") else self.model

            named = list(module.named_modules())
            idx = self.args.tta_feature_layer
            if isinstance(idx, str):
                target = dict(named).get(idx, None)
            elif isinstance(idx, int):
                idx = idx % len(named)
                target = named[idx][1]
            else:
                target = named[-2][1]
            if target is None:
                self.tta_enabled = False
                return

            target.register_forward_hook(lambda m, inp, out: setattr(self, "tta_features", out))

            try:
                model_path = (
                    Path(self.args.model)
                    if isinstance(self.args.model, (str, Path)) and Path(self.args.model).is_file()
                    else None
                )
                tta_stats_path = model_path.parent / "tta_stats.pt" if model_path else None

                if tta_stats_path is None or not tta_stats_path.exists():
                    train_dir = self.save_dir.parent / "train"
                    if train_dir.is_dir():
                        tta_stats_path = train_dir / "tta_stats.pt"
                    else:
                        tta_stats_path = self.save_dir / "tta_stats.pt"

                if tta_stats_path.exists():
                    self.tta_stats = torch.load(tta_stats_path, map_location=self.device)
                    LOGGER.info(f"Loaded TTA statistics from {tta_stats_path}")

                    default_D_in_KL = 0.5
                    D_in_KL = self.tta_stats.get("D_in_KL", default_D_in_KL)
                    if D_in_KL == default_D_in_KL and "D_in_KL" not in self.tta_stats:
                        LOGGER.warning(f"D_in_KL not found in TTA stats, using default value: {default_D_in_KL}")

                    mu_img = self.tta_stats.get("mu_img")
                    inv_sigma_sq_img = self.tta_stats.get("inv_sigma_sq_img")
                    mu_obj_dict = self.tta_stats.get("mu_obj_dict")
                    inv_sigma_sq_obj_dict = self.tta_stats.get("inv_sigma_sq_obj_dict")
                    pi_obj = self.tta_stats.get("pi_obj")

                    self.tta_criterion = FeatureAlignmentLoss(
                        nc=len(self.model.names),
                        feat_stats={
                            "mu_img": mu_img,
                            "inv_sigma_sq_img": inv_sigma_sq_img,
                            "mu_obj_dict": mu_obj_dict,
                            "inv_sigma_sq_obj_dict": inv_sigma_sq_obj_dict,
                            "pi_obj": pi_obj,
                        },
                        alpha=self.args.tta_alpha,
                        alpha_ema_loss=self.args.tta_alpha_ema_loss,
                        D_in_KL=D_in_KL,
                        tau1=self.args.tta_tau1,
                        tau2=self.args.tta_tau2,
                    ).to(self.device)
                    tta_params = []
                    target_model = module
                    for p in target_model.parameters():
                        p.requires_grad_(False)

                    LOGGER.info(f"Setting up TTA with update mode: '{self.args.tta_update_mode}'")

                    if self.args.tta_update_mode == "adaptor":
                        LOGGER.info("TTA mode: Finding Adaptor layers to update.")
                        for m in target_model.modules():
                            if isinstance(m, Adaptor):
                                for p in m.parameters():
                                    p.requires_grad_(True)
                                    tta_params.append(p)
                        if not tta_params:
                            LOGGER.warning("TTA update mode is 'adaptor', but no Adaptor layers found.")

                    elif self.args.tta_update_mode == "full":
                        LOGGER.info("TTA mode: Setting all model parameters to be updated.")
                        for p in target_model.parameters():
                            p.requires_grad_(True)
                            tta_params.append(p)  # Collect all parameters

                    else:
                        LOGGER.warning(
                            f"Unknown tta_update_mode: '{self.args.tta_update_mode}'. Supported modes are 'adaptor' and 'full'. Defaulting to no TTA updates."
                        )
                        # Ensure tta_params is empty if mode is unknown
                        tta_params = []

                    if not tta_params:
                        LOGGER.warning(
                            f"TTA enabled but no parameters selected for update mode '{self.args.tta_update_mode}'. Disabling TTA."
                        )
                        self.tta_enabled = False
                    else:
                        self.tta_optimizer = optim.Adam(tta_params, lr=self.args.tta_lr)
                        self.tta_enabled = True
                        LOGGER.info(
                            f"TTA enabled with {len(tta_params)} trainable parameters (mode='{self.args.tta_update_mode}', lr={self.args.tta_lr})"
                        )

                else:
                    LOGGER.warning(
                        f"TTA statistics file not found at expected locations ({tta_stats_path} or others). TTA disabled."
                    )
                    self.tta_enabled = False
            except Exception as e:
                LOGGER.error(f"Failed to setup TTA: {e}", exc_info=True)
                self.tta_enabled = False
        else:
            self.tta_enabled = False

    def write_results(self, i, p, im, s):
        """
        Write inference results to a file or directory.

        Args:
            i (int): Index of the current image in the batch.
            p (Path): Path to the current image.
            im (torch.Tensor): Preprocessed image tensor.
            s (List[str]): List of result strings.

        Returns:
            (str): String with result information.
        """
        string = ""  # print string
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        if self.source_type.stream or self.source_type.from_img or self.source_type.tensor:  # batch_size >= 1
            string += f"{i}: "
            frame = self.dataset.count
        else:
            match = re.search(r"frame (\d+)/", s[i])
            frame = int(match[1]) if match else None  # 0 if frame undetermined

        self.txt_path = self.save_dir / "labels" / (p.stem + ("" if self.dataset.mode == "image" else f"_{frame}"))
        string += "{:g}x{:g} ".format(*im.shape[2:])
        result = self.results[i]
        result.save_dir = self.save_dir.__str__()  # used in other locations
        string += f"{result.verbose()}{result.speed['inference']:.1f}ms"

        # Add predictions to image
        if self.args.save or self.args.show:
            self.plotted_img = result.plot(
                line_width=self.args.line_width,
                boxes=self.args.show_boxes,
                conf=self.args.show_conf,
                labels=self.args.show_labels,
                im_gpu=None if self.args.retina_masks else im[i],
            )

        # Save results
        if self.args.save_txt:
            result.save_txt(f"{self.txt_path}.txt", save_conf=self.args.save_conf)
        if self.args.save_crop:
            result.save_crop(save_dir=self.save_dir / "crops", file_name=self.txt_path.stem)
        if self.args.show:
            self.show(str(p))
        if self.args.save:
            self.save_predicted_images(str(self.save_dir / p.name), frame)

        return string

    def save_predicted_images(self, save_path="", frame=0):
        """
        Save video predictions as mp4 or images as jpg at specified path.

        Args:
            save_path (str): Path to save the results.
            frame (int): Frame number for video mode.
        """
        im = self.plotted_img

        # Save videos and streams
        if self.dataset.mode in {"stream", "video"}:
            fps = self.dataset.fps if self.dataset.mode == "video" else 30
            frames_path = f"{save_path.split('.', 1)[0]}_frames/"
            if save_path not in self.vid_writer:  # new video
                if self.args.save_frames:
                    Path(frames_path).mkdir(parents=True, exist_ok=True)
                suffix, fourcc = (".mp4", "avc1") if MACOS else (".avi", "WMV2") if WINDOWS else (".avi", "MJPG")
                self.vid_writer[save_path] = cv2.VideoWriter(
                    filename=str(Path(save_path).with_suffix(suffix)),
                    fourcc=cv2.VideoWriter_fourcc(*fourcc),
                    fps=fps,  # integer required, floats produce error in MP4 codec
                    frameSize=(im.shape[1], im.shape[0]),  # (width, height)
                )

            # Save video
            self.vid_writer[save_path].write(im)
            if self.args.save_frames:
                cv2.imwrite(f"{frames_path}{frame}.jpg", im)

        # Save images
        else:
            cv2.imwrite(str(Path(save_path).with_suffix(".jpg")), im)  # save to JPG for best support

    def show(self, p=""):
        """Display an image in a window."""
        im = self.plotted_img
        if platform.system() == "Linux" and p not in self.windows:
            self.windows.append(p)
            cv2.namedWindow(p, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            cv2.resizeWindow(p, im.shape[1], im.shape[0])  # (width, height)
        cv2.imshow(p, im)
        cv2.waitKey(300 if self.dataset.mode == "image" else 1)  # 1 millisecond

    def run_callbacks(self, event: str):
        """Run all registered callbacks for a specific event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def add_callback(self, event: str, func):
        """Add a callback function for a specific event."""
        self.callbacks[event].append(func)
