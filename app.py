#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv12 Interactive Demo with Test-Time Adaptation
=================================================
A browser-based GUI for YOLOv12 object detection with TTA support.
Allows users to upload images/videos and run inference with configurable parameters.

Author: Ultralytics
License: AGPL-3.0
"""

import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import logging

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("YOLOv12-TTA")

# Constants
ROOT = Path(__file__).resolve().parent
ASSETS_DIR = ROOT / "ultralytics" / "assets"
DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IMG_SIZE = 640
TEMP_DIR = Path(tempfile.gettempdir()) / "yolov12_demo"

# Supported models
MODEL_VARIANTS = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
    "yolov12n.pt",
    "yolov12s.pt",
    "yolov12m.pt",
    "yolov12l.pt",
    "yolov12x.pt",
]

# Example image filenames
EXAMPLE_IMAGES = ["bus.jpg", "zidane.jpg"]


def ensure_dir_exists(directory: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist and return Path object"""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_asset_path(filename: str) -> str:
    """Get asset file path, supporting fallback paths"""
    # Try different possible paths
    possible_paths = [
        ASSETS_DIR / filename,  # Main path
        ROOT / "ultralytics" / "assets" / filename,  # Fallback 1
        ROOT / "assets" / filename,  # Fallback 2
    ]

    for path in possible_paths:
        if path.exists():
            return str(path)

    # If all paths fail, log warning and return a potentially valid default path
    logger.warning(f"Asset file not found: {filename}")
    return str(possible_paths[0])


class YOLOv12TTA:
    """YOLOv12 Test-Time Adaptation inference manager."""

    def __init__(self) -> None:
        """Initialize inference environment"""
        # Create temporary directory
        self.temp_dir = ensure_dir_exists(TEMP_DIR)
        self.models_cache: Dict[str, YOLO] = {}  # Model cache
        logger.info(f"Initializing YOLOv12-TTA engine, temp dir: {self.temp_dir}")

    def get_model(
        self,
        model_id: str,
        use_tta: bool = False,
        tta_alpha: float = 0.01,
        tta_tau1: float = 1.1,
        tta_tau2: float = 1.05,
        tta_momentum: float = 0.99,
    ) -> YOLO:
        """Get or create model instance with TTA configuration"""
        # Create unique key for model configuration
        config_key = f"{model_id}_tta{use_tta}_{tta_alpha}_{tta_tau1}_{tta_tau2}_{tta_momentum}"

        # If model already cached, return it directly
        if config_key in self.models_cache:
            logger.debug(f"Using cached model: {model_id}")
            return self.models_cache[config_key]

        try:
            # Create new model instance
            logger.info(f"Loading model: {model_id}")
            model = YOLO(model_id)

            # Configure TTA if needed
            if use_tta:
                logger.info(
                    f"Enabling TTA with params: α={tta_alpha}, τ1={tta_tau1}, τ2={tta_tau2}, momentum={tta_momentum}"
                )
                model.enable_tta()
                model.configure_tta(alpha=tta_alpha, tau1=tta_tau1, tau2=tta_tau2, momentum=tta_momentum)

            # Cache and return model
            self.models_cache[config_key] = model
            return model

        except Exception as e:
            logger.error(f"Model loading failed: {model_id}, error: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def run_inference(
        self,
        image: Optional[Image.Image],
        video: Optional[str],
        model_id: str,
        image_size: int,
        conf_threshold: float,
        use_tta: bool = False,
        tta_alpha: float = 0.01,
        tta_tau1: float = 1.1,
        tta_tau2: float = 1.05,
        tta_momentum: float = 0.99,
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Run YOLOv12 inference on image or video with optional TTA

        Returns:
            (annotated image, output video path)
        """
        try:
            # Load model
            start_time = time.time()
            model = self.get_model(model_id, use_tta, tta_alpha, tta_tau1, tta_tau2, tta_momentum)

            # Process image input
            if image is not None:
                return self._process_image(image, model, image_size, conf_threshold, use_tta, start_time), None

            # Process video input
            elif video:
                return None, self._process_video(video, model, image_size, conf_threshold, use_tta)

            else:
                return None, None

        except Exception as e:
            logger.error(f"Error during inference: {e}")
            import traceback

            traceback.print_exc()
            # Return empty image on error
            return np.zeros((400, 600, 3), dtype=np.uint8), None

    def _process_image(
        self,
        image: Image.Image,
        model: YOLO,
        image_size: int,
        conf_threshold: float,
        use_tta: bool,
        start_time: float,
    ) -> np.ndarray:
        """Process single image and return annotated image"""
        # Run prediction
        results = model.predict(source=image, imgsz=image_size, conf=conf_threshold, verbose=False)

        if not results or len(results) == 0:
            return np.zeros((400, 600, 3), dtype=np.uint8)

        # Get annotated image
        annotated_image = results[0].plot()
        process_time = time.time() - start_time
        detected_objects = len(results[0].boxes)

        # Add time and detection info
        cv2.putText(
            annotated_image,
            f"Time: {process_time:.3f}s | Detections: {detected_objects}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        if use_tta:
            cv2.putText(annotated_image, "TTA Enabled", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Convert BGR to RGB for correct display in Gradio
        return annotated_image[:, :, ::-1]

    def _process_video(
        self,
        video: str,
        model: YOLO,
        image_size: int,
        conf_threshold: float,
        use_tta: bool,
    ) -> str:
        """Process video and return path to processed video"""
        # Create temporary file
        timestamp = int(time.time())
        video_path = TEMP_DIR / f"input_{timestamp}.webm"

        # Copy uploaded video to temp dir
        with open(video_path, "wb") as f:
            with open(video, "rb") as g:
                f.write(g.read())

        # Get video properties
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Prepare output video
        output_video_path = TEMP_DIR / f"output_{timestamp}.webm"
        out = cv2.VideoWriter(str(output_video_path), cv2.VideoWriter_fourcc(*"vp80"), fps, (frame_width, frame_height))

        # Process frames
        logger.info(f"Starting video processing, output path: {output_video_path}")
        frame_count = 0
        total_objects = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            results = model.predict(source=frame, imgsz=image_size, conf=conf_threshold, verbose=False)
            annotated_frame = results[0].plot()
            total_objects += len(results[0].boxes)

            # Add info to frame
            cv2.putText(
                annotated_frame,
                f"Frame: {frame_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            if use_tta:
                cv2.putText(annotated_frame, "TTA Enabled", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            out.write(annotated_frame)

        # Clean up resources
        cap.release()
        out.release()
        logger.info(f"Video processing complete, {frame_count} frames, {total_objects} detected objects")

        # Delete temporary input file
        try:
            os.remove(video_path)
        except OSError:
            logger.warning(f"Could not delete temporary file {video_path}")

        return str(output_video_path)


class GradioInterface:
    """Gradio interface for YOLOv12 TTA demo"""

    def __init__(self) -> None:
        """Initialize Gradio interface"""
        self.tta_engine = YOLOv12TTA()
        # Check if example images exist
        self.example_assets = []
        for img_name in EXAMPLE_IMAGES:
            self.example_assets.append(get_asset_path(img_name))

    def run_inference(
        self,
        image: Optional[Image.Image],
        video: Optional[str],
        model_id: str,
        image_size: int,
        conf_threshold: float,
        input_type: str,
        use_tta: bool,
        tta_alpha: float,
        tta_tau1: float,
        tta_tau2: float,
        tta_momentum: float,
    ) -> Tuple[Optional[np.ndarray], Optional[str], str, int]:
        """Run inference with selected parameters and return results"""
        start_time = time.time()

        if input_type == "Image":
            img_output, vid_output = self.tta_engine.run_inference(
                image, None, model_id, image_size, conf_threshold, use_tta, tta_alpha, tta_tau1, tta_tau2, tta_momentum
            )
            process_time = time.time() - start_time

            # Calculate detection count if available
            detection_num = 0
            if image is not None:
                try:
                    model = self.tta_engine.get_model(model_id, use_tta, tta_alpha, tta_tau1, tta_tau2, tta_momentum)
                    results = model.predict(source=image, imgsz=image_size, conf=conf_threshold)
                    detection_num = len(results[0].boxes)
                except Exception as e:
                    logger.error(f"Error calculating detection count: {e}")

            return img_output, vid_output, f"{process_time:.3f} seconds", detection_num
        else:
            img_output, vid_output = self.tta_engine.run_inference(
                None, video, model_id, image_size, conf_threshold, use_tta, tta_alpha, tta_tau1, tta_tau2, tta_momentum
            )
            process_time = time.time() - start_time
            return img_output, vid_output, f"{process_time:.3f} seconds", 0

    def run_inference_for_examples(
        self, image: Image.Image, model_path: str, image_size: int, conf_threshold: float, use_tta: bool = False
    ) -> np.ndarray:
        """Run inference on example images with optional TTA"""
        annotated_image, _ = self.tta_engine.run_inference(image, None, model_path, image_size, conf_threshold, use_tta)
        return annotated_image

    @staticmethod
    def update_visibility(input_type: str) -> Tuple:
        """Toggle visibility based on input type"""
        image_visible = input_type == "Image"
        return (
            gr.Image(visible=image_visible),
            gr.Video(visible=not image_visible),
            gr.Image(visible=image_visible),
            gr.Video(visible=not image_visible),
        )

    @staticmethod
    def toggle_tta_params(use_tta: bool) -> gr.Column:
        """Show/hide TTA parameters based on TTA checkbox"""
        return gr.Column(visible=use_tta)

    def create_interface(self) -> gr.Blocks:
        """Create and configure Gradio application interface"""
        with gr.Blocks(css="footer {visibility: hidden}") as demo:
            # Model configuration section
            with gr.Row():
                # Left column - Input & Model selection
                with gr.Column(scale=1):
                    # Input selection
                    with gr.Group():
                        gr.Markdown("### Input")
                        input_type = gr.Radio(
                            choices=["Image", "Video"],
                            value="Image",
                            label="Input Type",
                            info="Select input type for object detection",
                        )
                        image = gr.Image(type="pil", label="Upload Image", visible=True)
                        video = gr.Video(label="Upload Video", visible=False)

                    # Model selection
                    with gr.Group():
                        gr.Markdown("### Model")
                        model_id = gr.Dropdown(
                            label="Select Model",
                            choices=MODEL_VARIANTS,
                            value=DEFAULT_MODEL,
                            info="Choose YOLO model variant",
                        )

                    # Detection parameters
                    with gr.Group():
                        gr.Markdown("### Detection Parameters")
                        image_size = gr.Slider(
                            label="Image Size",
                            minimum=320,
                            maximum=1280,
                            step=32,
                            value=DEFAULT_IMG_SIZE,
                            info="Input resolution for the model",
                        )
                        conf_threshold = gr.Slider(
                            label="Confidence Threshold",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.05,
                            value=DEFAULT_CONF_THRESHOLD,
                            info="Minimum detection confidence score",
                        )

                    # Run detection button
                    yolov12_infer = gr.Button(value="Detect Objects")

                # Right column - Results & TTA options
                with gr.Column(scale=1):
                    # Results section
                    with gr.Group():
                        gr.Markdown("### Detection Results")
                        output_image = gr.Image(type="numpy", label="Detected Objects", visible=True)
                        output_video = gr.Video(label="Processed Video", visible=False)

                        with gr.Row():
                            with gr.Column(scale=1):
                                inference_time = gr.Textbox(
                                    label="Inference Time", value="", info="Model processing time"
                                )
                            with gr.Column(scale=1):
                                detection_count = gr.Number(
                                    label="Detection Count", value=0, info="Number of detected objects"
                                )

                    # TTA parameters - Moved to right column
                    with gr.Group():
                        gr.Markdown("### Test-Time Adaptation (TTA)")
                        use_tta = gr.Checkbox(
                            label="Enable Test-Time Adaptation",
                            value=False,
                            info="Enable dynamic model adaptation during inference",
                        )
                        with gr.Column(visible=False) as tta_params:
                            tta_alpha = gr.Slider(
                                label="TTA Alpha (Learning Rate)",
                                minimum=0.001,
                                maximum=0.1,
                                step=0.001,
                                value=0.01,
                                info="Controls model adaptation speed",
                            )
                            tta_tau1 = gr.Slider(
                                label="TTA Tau1 (Primary Threshold)",
                                minimum=1.0,
                                maximum=2.0,
                                step=0.05,
                                value=1.1,
                                info="Threshold for primary distribution shift",
                            )
                            tta_tau2 = gr.Slider(
                                label="TTA Tau2 (Secondary Threshold)",
                                minimum=1.0,
                                maximum=1.5,
                                step=0.01,
                                value=1.05,
                                info="Threshold for secondary distribution shift",
                            )
                            tta_momentum = gr.Slider(
                                label="TTA Momentum",
                                minimum=0.9,
                                maximum=0.999,
                                step=0.001,
                                value=0.99,
                                info="Momentum for EMA updates",
                            )

            # Set up event handlers
            use_tta.change(fn=self.toggle_tta_params, inputs=[use_tta], outputs=[tta_params])
            input_type.change(
                fn=self.update_visibility, inputs=[input_type], outputs=[image, video, output_image, output_video]
            )
            yolov12_infer.click(
                fn=self.run_inference,
                inputs=[
                    image,
                    video,
                    model_id,
                    image_size,
                    conf_threshold,
                    input_type,
                    use_tta,
                    tta_alpha,
                    tta_tau1,
                    tta_tau2,
                    tta_momentum,
                ],
                outputs=[output_image, output_video, inference_time, detection_count],
            )

            # Example images (with and without TTA comparison)
            with gr.Row():
                gr.Examples(
                    examples=[
                        [self.example_assets[0], "yolov8s.pt", 640, 0.25, False],
                        [self.example_assets[1], "yolov8n.pt", 640, 0.25, False],
                        [self.example_assets[0], "yolov8s.pt", 640, 0.25, True],
                    ],
                    fn=self.run_inference_for_examples,
                    inputs=[image, model_id, image_size, conf_threshold, use_tta],
                    outputs=output_image,
                    cache_examples=False,  # Fixed: Disable caching to avoid errors
                )

        return demo


def main() -> None:
    """Application entry point"""
    # Ensure temp directory exists
    ensure_dir_exists(TEMP_DIR)
    logger.info(f"Starting YOLOv12-TTA demo, temp dir: {TEMP_DIR}")

    # Set up custom styling for the Gradio app
    css = """
        .gradio-container {max-width: 1200px; margin: 0 auto;}
        .gr-box {border-radius: 10px; border: 1px solid #e0e0e0;}
        .gr-button {background-color: #1565c0;}
        .gr-button:hover {background-color: #0d47a1;}
        .footer {visibility: hidden}
    """

    # Create interface
    interface = GradioInterface()

    with gr.Blocks(css=css) as gradio_app:
        # Title section
        gr.HTML(
            """
            <div style="text-align: center; margin-bottom: 10px">
                <img src="https://ultralytics.com/images/yolov8-logo.png" 
                     alt="YOLOv12 Logo" style="height: 60px;">
            </div>
            <h1 style="text-align: center; margin-bottom: 10px">
                YOLOv12: Attention-based Real-time Object Detection with Test-Time Adaptation
            </h1>
            """
        )

        # Links section
        gr.HTML(
            """
            <div style="text-align: center; margin-bottom: 20px">
                <a href="https://arxiv.org/abs/2502.12524" target="_blank" 
                   style="text-decoration: none; padding: 5px 10px; margin: 0 10px; 
                          border: 1px solid #ddd; border-radius: 5px; color: #1565c0;">
                    <span style="font-size: 1.2em">📄 Paper</span>
                </a>
                <a href="https://github.com/sunsmarterjie/yolov12" target="_blank" 
                   style="text-decoration: none; padding: 5px 10px; margin: 0 10px; 
                          border: 1px solid #ddd; border-radius: 5px; color: #1565c0;">
                    <span style="font-size: 1.2em">💻 GitHub</span>
                </a>
                <a href="https://ultralytics.com" target="_blank" 
                   style="text-decoration: none; padding: 5px 10px; margin: 0 10px; 
                          border: 1px solid #ddd; border-radius: 5px; color: #1565c0;">
                    <span style="font-size: 1.2em">🚀 Ultralytics</span>
                </a>
            </div>
            """
        )

        # Description
        gr.HTML(
            """
            <div style="text-align: center; margin-bottom: 20px; padding: 0 50px">
                <p>This demo showcases YOLOv12 object detection with Test-Time Adaptation (TTA).
                   TTA allows the model to adapt to new environments during inference without retraining.</p>
            </div>
            """
        )

        # Main program
        interface.create_interface()

        # Footer
        gr.HTML(
            """
            <div style="text-align: center; margin-top: 20px; padding: 10px; font-size: 0.8em; color: #666">
                <p>Built with <a href="https://gradio.app">Gradio</a> and 
                   <a href="https://github.com/ultralytics/ultralytics">Ultralytics YOLOv12</a></p>
            </div>
            """
        )

    try:
        # Launch the app
        logger.info("Starting Gradio web interface")
        gradio_app.launch(
            share=False,  # Set to True to create a public link
            debug=False,  # Set to True during development
            inbrowser=True,  # Auto-open in browser
        )
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Error launching Gradio interface: {e}")


if __name__ == "__main__":
    main()
