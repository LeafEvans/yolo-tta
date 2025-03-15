#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv12-TTA Performance Benchmark
=================================
Test the performance difference between standard YOLOv12 and YOLOv12 with TTA across different scenarios.

Usage:
    python benchmark.py --model yolov8n.pt --data coco.yaml --batch-size 32
    python benchmark.py --model yolov8n.pt --scenario domain_shift --output benchmark_results
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt

from ultralytics import YOLO
from ultralytics.utils import LOGGER, colorstr

# Define test scenarios
SCENARIOS = {
    "standard": "Standard dataset testing",
    "domain_shift": "Domain shift testing (e.g., day to night)",
    "weather": "Different weather conditions (rain, snow, fog)",
    "noise": "Noise robustness testing",
    "blur": "Blur and motion blur testing",
    "all": "Run all test scenarios",
}


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="YOLOv12-TTA Performance Benchmark")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Model path")
    parser.add_argument("--data", type=str, default="coco.yaml", help="Dataset config file")
    parser.add_argument("--batch-size", type=int, default=32, help="Test batch size")
    parser.add_argument("--img-size", type=int, default=640, help="Input image size")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument(
        "--scenario", type=str, default="standard", help=f"Test scenario: {', '.join(SCENARIOS.keys())}"
    )
    parser.add_argument("--tta-alpha", type=float, default=0.01, help="TTA learning rate")
    parser.add_argument("--tta-tau1", type=float, default=1.1, help="Primary distribution shift threshold")
    parser.add_argument("--tta-tau2", type=float, default=1.05, help="Secondary distribution shift threshold")
    parser.add_argument("--tta-momentum", type=float, default=0.99, help="TTA momentum coefficient")
    parser.add_argument("--device", default="", help="CUDA device, e.g. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--output", type=str, default="tta_benchmark", help="Output directory")
    parser.add_argument("--visualize", action="store_true", help="Visualize detection results")
    parser.add_argument("--seed", type=int, default=17, help="Random seed")
    return parser.parse_args()


def apply_transforms(image, scenario):
    """Apply image transformations based on different scenarios"""
    if scenario == "standard":
        return image  # No transformation

    elif scenario == "domain_shift":
        # Simulate day-to-night transformation
        image = np.array(image).astype(np.float32) * 0.5
        return np.clip(image, 0, 255).astype(np.uint8)

    elif scenario == "weather":
        # Simulate foggy effect
        image = np.array(image).astype(np.float32)
        fog = np.ones_like(image) * 200
        image = image * 0.7 + fog * 0.3
        return np.clip(image, 0, 255).astype(np.uint8)

    elif scenario == "noise":
        # Add Gaussian noise
        image = np.array(image).astype(np.float32)
        noise = np.random.normal(0, 20, image.shape)
        image = image + noise
        return np.clip(image, 0, 255).astype(np.uint8)

    elif scenario == "blur":
        # Apply Gaussian blur
        image = np.array(image)
        return cv2.GaussianBlur(image, (15, 15), 0)

    return image


def run_benchmark(args):
    """Run performance benchmark tests"""
    # Set up output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load dataset configuration
    with open(args.data, errors="ignore") as f:
        data = yaml.safe_load(f)

    # Determine test scenarios
    scenarios = [args.scenario] if args.scenario != "all" else list(SCENARIOS.keys())[:-1]

    results = {}

    for scenario in scenarios:
        LOGGER.info(f"\n{colorstr('blue', 'bold', f'Scenario: {scenario} - {SCENARIOS[scenario]}')}")

        # Standard model (no TTA)
        LOGGER.info(f"\n{colorstr('green', 'bold', 'Testing standard model (no TTA)')}")
        model_std = YOLO(args.model)

        # TTA model
        LOGGER.info(f"\n{colorstr('green', 'bold', 'Testing TTA model')}")
        model_tta = YOLO(args.model)
        model_tta.enable_tta()
        model_tta.configure_tta(
            alpha=args.tta_alpha, tau1=args.tta_tau1, tau2=args.tta_tau2, momentum=args.tta_momentum
        )

        # Prepare validation dataset path
        val_path = data["val"]

        # Load image list
        val_images = []
        if os.path.isdir(val_path):
            for ext in ["jpg", "jpeg", "png", "bmp"]:
                val_images.extend(list(Path(val_path).rglob(f"*.{ext}")))
        else:
            # Assume it's a COCO format annotation file
            import json

            with open(val_path, "r") as f:
                annotations = json.load(f)
            img_dir = Path(val_path).parent / "images"
            val_images = [img_dir / img["file_name"] for img in annotations["images"]]

        # Randomly select a subset of images for testing
        if len(val_images) > 200:
            val_images = np.random.choice(val_images, 200, replace=False).tolist()

        # Test results
        results[scenario] = {"standard": {"metrics": {}, "times": []}, "tta": {"metrics": {}, "times": []}}

        # Create scenario-specific output directory
        scenario_dir = output_dir / scenario
        scenario_dir.mkdir(exist_ok=True)

        # Run standard model validation
        std_results = model_std.val(
            data=args.data,
            batch=args.batch_size,
            imgsz=args.img_size,
            conf=args.conf_thres,
            iou=args.iou_thres,
            device=args.device,
        )
        results[scenario]["standard"]["metrics"] = {
            "mAP50": float(std_results.box.map50),
            "mAP50-95": float(std_results.box.map),
            "precision": float(std_results.box.mp),
            "recall": float(std_results.box.mr),
        }

        # Run TTA model validation
        tta_results = model_tta.val(
            data=args.data,
            batch=args.batch_size,
            imgsz=args.img_size,
            conf=args.conf_thres,
            iou=args.iou_thres,
            device=args.device,
        )
        results[scenario]["tta"]["metrics"] = {
            "mAP50": float(tta_results.box.map50),
            "mAP50-95": float(tta_results.box.map),
            "precision": float(tta_results.box.mp),
            "recall": float(tta_results.box.mr),
        }

        # Additional single-image testing (simulating real usage)
        LOGGER.info(f"{colorstr('blue', 'Single image inference testing')}")

        # Create visualization directory
        vis_dir = scenario_dir / "visualization"
        if args.visualize:
            vis_dir.mkdir(exist_ok=True)

        # Test single image inference performance and results
        for i, img_path in enumerate(tqdm(val_images[:20], desc="Testing image inference")):
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Apply scenario transformations
            img_transformed = apply_transforms(img, scenario)

            # Standard model inference
            t0 = time.time()
            std_pred = model_std.predict(
                source=img_transformed, imgsz=args.img_size, conf=args.conf_thres, iou=args.iou_thres, verbose=False
            )
            std_time = time.time() - t0
            results[scenario]["standard"]["times"].append(std_time)

            # TTA model inference
            t0 = time.time()
            tta_pred = model_tta.predict(
                source=img_transformed, imgsz=args.img_size, conf=args.conf_thres, iou=args.iou_thres, verbose=False
            )
            tta_time = time.time() - t0
            results[scenario]["tta"]["times"].append(tta_time)

            # Visualize comparison
            if args.visualize:
                std_img = std_pred[0].plot()
                tta_img = tta_pred[0].plot()

                # Add titles
                cv2.putText(
                    std_img,
                    f"Standard: {len(std_pred[0].boxes)} objects, {std_time:.3f}s",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    tta_img,
                    f"TTA: {len(tta_pred[0].boxes)} objects, {tta_time:.3f}s",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

                # Merge two images
                h, w = std_img.shape[:2]
                comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
                comparison[:, :w, :] = std_img
                comparison[:, w:, :] = tta_img

                # Save comparison image
                cv2.imwrite(str(vis_dir / f"comparison_{i}.jpg"), comparison)

        # Calculate average inference time
        if results[scenario]["standard"]["times"]:
            results[scenario]["standard"]["avg_time"] = np.mean(results[scenario]["standard"]["times"])
            results[scenario]["tta"]["avg_time"] = np.mean(results[scenario]["tta"]["times"])

        # Results summary
        LOGGER.info(f"\n{colorstr('green', 'bold', f'{scenario} scenario results summary:')}")
        LOGGER.info(
            f"Standard model: mAP50={results[scenario]['standard']['metrics']['mAP50']:.4f}, "
            f"Avg inference time: {results[scenario]['standard'].get('avg_time', 0):.4f}s"
        )
        LOGGER.info(
            f"TTA model: mAP50={results[scenario]['tta']['metrics']['mAP50']:.4f}, "
            f"Avg inference time: {results[scenario]['tta'].get('avg_time', 0):.4f}s"
        )

    # Save results to JSON
    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)

    # Generate performance comparison charts
    generate_charts(results, output_dir)

    # Generate Markdown format table results
    generate_markdown_table(results, output_dir, args)

    return results


def generate_charts(results, output_dir):
    """Generate performance comparison charts"""
    plt.style.use("ggplot")
    scenarios = list(results.keys())

    # mAP50 comparison chart
    plt.figure(figsize=(12, 6))
    std_map = [results[s]["standard"]["metrics"]["mAP50"] for s in scenarios]
    tta_map = [results[s]["tta"]["metrics"]["mAP50"] for s in scenarios]

    x = np.arange(len(scenarios))
    width = 0.35

    plt.bar(x - width / 2, std_map, width, label="Standard", color="#2C7BB6")
    plt.bar(x + width / 2, tta_map, width, label="TTA", color="#D7191C")

    plt.ylabel("mAP@0.5")
    plt.title("YOLOv12 vs YOLOv12-TTA mAP Comparison")
    plt.xticks(x, [SCENARIOS[s] for s in scenarios], rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "map_comparison.png", dpi=300)

    # Precision-Recall comparison chart
    plt.figure(figsize=(12, 6))

    # Split into two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Precision chart
    std_precision = [results[s]["standard"]["metrics"]["precision"] for s in scenarios]
    tta_precision = [results[s]["tta"]["metrics"]["precision"] for s in scenarios]

    ax1.bar(x - width / 2, std_precision, width, label="Standard", color="#2C7BB6")
    ax1.bar(x + width / 2, tta_precision, width, label="TTA", color="#D7191C")
    ax1.set_ylabel("Precision")
    ax1.set_title("Precision Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels([s[:10] for s in scenarios], rotation=30, ha="right")
    ax1.legend()

    # Recall chart
    std_recall = [results[s]["standard"]["metrics"]["recall"] for s in scenarios]
    tta_recall = [results[s]["tta"]["metrics"]["recall"] for s in scenarios]

    ax2.bar(x - width / 2, std_recall, width, label="Standard", color="#2C7BB6")
    ax2.bar(x + width / 2, tta_recall, width, label="TTA", color="#D7191C")
    ax2.set_ylabel("Recall")
    ax2.set_title("Recall Comparison")
    ax2.set_xticks(x)
    ax2.set_xticklabels([s[:10] for s in scenarios], rotation=30, ha="right")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_comparison.png", dpi=300)

    # Inference time comparison chart
    plt.figure(figsize=(12, 6))

    std_time = [results[s]["standard"].get("avg_time", 0) for s in scenarios]
    tta_time = [results[s]["tta"].get("avg_time", 0) for s in scenarios]

    plt.bar(x - width / 2, std_time, width, label="Standard", color="#2C7BB6")
    plt.bar(x + width / 2, tta_time, width, label="TTA", color="#D7191C")

    plt.ylabel("Inference Time (seconds)")
    plt.title("Inference Speed Comparison")
    plt.xticks(x, [SCENARIOS[s] for s in scenarios], rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "inference_time_comparison.png", dpi=300)

    # Performance improvement rate chart
    plt.figure(figsize=(12, 6))

    map_improvement = [(tta - std) / std * 100 for tta, std in zip(tta_map, std_map)]
    time_overhead = [(tta - std) / std * 100 for tta, std in zip(tta_time, std_time)]

    plt.bar(x - width / 2, map_improvement, width, label="mAP Improvement (%)", color="#33a02c")
    plt.bar(x + width / 2, time_overhead, width, label="Time Overhead Increase (%)", color="#ff7f00")

    plt.ylabel("Change Rate (%)")
    plt.title("TTA Performance Change Relative to Standard Model")
    plt.xticks(x, [SCENARIOS[s] for s in scenarios], rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "performance_change.png", dpi=300)


def generate_markdown_table(results, output_dir, args):
    """Generate Markdown format table results"""
    with open(output_dir / "benchmark_table.md", "w") as f:
        f.write("# YOLOv12-TTA Performance Comparison Results\n\n")

        # Model configuration information
        f.write("## Test Configuration\n\n")
        f.write(f"- **Model**: {args.model}\n")
        f.write(
            f"- **TTA Parameters**: alpha={args.tta_alpha}, tau1={args.tta_tau1}, tau2={args.tta_tau2}, momentum={args.tta_momentum}\n"
        )
        f.write(f"- **Test Resolution**: {args.img_size}x{args.img_size}\n")
        f.write(f"- **Confidence Threshold**: {args.conf_thres}\n")
        f.write(f"- **IoU Threshold**: {args.iou_thres}\n\n")

        # Main performance table
        f.write("## Performance Comparison Table\n\n")
        f.write("| Scenario | Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Inference Time(s) |\n")
        f.write("|----------|-------|---------|--------------|-----------|--------|-------------------|\n")

        for scenario in results:
            # Standard model row
            std_metrics = results[scenario]["standard"]["metrics"]
            std_time = results[scenario]["standard"].get("avg_time", 0)

            f.write(
                f"| {SCENARIOS[scenario]} | Standard | {std_metrics['mAP50']:.4f} | {std_metrics['mAP50-95']:.4f} | {std_metrics['precision']:.4f} | {std_metrics['recall']:.4f} | {std_time:.4f} |\n"
            )

            # TTA model row
            tta_metrics = results[scenario]["tta"]["metrics"]
            tta_time = results[scenario]["tta"].get("avg_time", 0)

            f.write(
                f"| {SCENARIOS[scenario]} | **TTA** | **{tta_metrics['mAP50']:.4f}** | **{tta_metrics['mAP50-95']:.4f}** | **{tta_metrics['precision']:.4f}** | **{tta_metrics['recall']:.4f}** | {tta_time:.4f} |\n"
            )

        # Performance improvement table
        f.write("\n## Performance Improvement Rate\n\n")
        f.write(
            "| Scenario | mAP@0.5 Improvement | mAP@0.5:0.95 Improvement | Precision Improvement | Recall Improvement | Time Overhead Increase |\n"
        )
        f.write(
            "|----------|--------------------|--------------------------|----------------------|-------------------|------------------------|\n"
        )

        for scenario in results:
            std_metrics = results[scenario]["standard"]["metrics"]
            tta_metrics = results[scenario]["tta"]["metrics"]
            std_time = results[scenario]["standard"].get("avg_time", 0)
            tta_time = results[scenario]["tta"].get("avg_time", 0)

            map50_imp = (tta_metrics["mAP50"] - std_metrics["mAP50"]) / std_metrics["mAP50"] * 100
            map_imp = (tta_metrics["mAP50-95"] - std_metrics["mAP50-95"]) / std_metrics["mAP50-95"] * 100
            prec_imp = (tta_metrics["precision"] - std_metrics["precision"]) / std_metrics["precision"] * 100
            recall_imp = (tta_metrics["recall"] - std_metrics["recall"]) / std_metrics["recall"] * 100
            time_overhead = (tta_time - std_time) / std_time * 100 if std_time > 0 else 0

            f.write(
                f"| {SCENARIOS[scenario]} | {map50_imp:+.2f}% | {map_imp:+.2f}% | {prec_imp:+.2f}% | {recall_imp:+.2f}% | {time_overhead:+.2f}% |\n"
            )

        # Chart references
        f.write("\n## Performance Comparison Charts\n\n")
        f.write("![mAP Comparison](./map_comparison.png)\n\n")
        f.write("![Precision-Recall Comparison](./precision_recall_comparison.png)\n\n")
        f.write("![Inference Time Comparison](./inference_time_comparison.png)\n\n")
        f.write("![Performance Change Comparison](./performance_change.png)\n")


def main(args):
    """Main function"""
    LOGGER.info(f"{colorstr('cyan', 'bold', 'YOLOv12-TTA Performance Benchmark')}")
    LOGGER.info(f"Test model: {args.model}")
    LOGGER.info(f"Test scenario: {args.scenario} - {SCENARIOS.get(args.scenario, 'All scenarios')}")

    # Check model file
    if not os.path.exists(args.model) and not args.model.startswith(("yolov8", "yolov12")):
        LOGGER.error(f"Model file not found: {args.model}")
        return

    # Run tests
    run_benchmark(args)

    LOGGER.info(f"\n{colorstr('green', 'bold', 'Testing complete!')}")
    LOGGER.info(f"Results saved to {args.output}/benchmark_results.json")
    LOGGER.info(f"Markdown table saved to {args.output}/benchmark_table.md")


if __name__ == "__main__":
    args = parse_args()
    main(args)
