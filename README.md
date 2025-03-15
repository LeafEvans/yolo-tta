# YOLOv12-TTA

<div align="center">
<p>
  <b>English</b> | <a href="READMD.zh-CN.md">简体中文</a>
</p>

<h2>YOLOv12: Attention-Centric Real-Time Object Detection with Test-Time Adaptation</h2>

Author: <a href="mailto:leafevans@foxmail.com">LeafEvans</a>

An enhanced implementation based on the original YOLOv12 work

</div>

## 🔥 Project Introduction

YOLOv12-TTA adds **Test-Time Adaptation** capability to YOLOv12, enabling the model to adapt to changing environments in real-time without retraining. This technology is particularly useful in:

- Changing environmental conditions (day/night, sunny/rainy)
- Camera angle and position variations
- Dramatic lighting fluctuations
- Maintaining detection accuracy in unseen environments

## ⚡ TTA Core Advantages

- **Continuous Learning**: Model adapts to new environments during inference
- **No Retraining Required**: Avoids time-consuming domain adaptation training
- **Dynamic Adjustment**: Automatically determines when adaptation is needed based on detection confidence
- **Lightweight**: Additional computational overhead less than 5%, minimal impact on real-time performance

## 📊 Performance

| Scenario                 | Standard YOLOv12n | With TTA | Improvement |
| ------------------------ | ----------------- | -------- | ----------- |
| Standard Conditions      | 40.4 mAP          | 40.7 mAP | +0.7%       |
| Domain Shift (Day→Night) | 28.3 mAP          | 35.6 mAP | **+25.8%**  |
| Weather Change (Fog)     | 30.1 mAP          | 36.2 mAP | **+20.3%**  |
| Added Noise              | 32.5 mAP          | 37.3 mAP | **+14.8%**  |
| Motion Blur              | 31.7 mAP          | 36.8 mAP | **+16.1%**  |

\*Tests performed on COCO validation images with synthetically applied transformations

## 📦 Installation Guide

```bash
# Install dependencies
pip install -r requirements.txt

# If flash-attention acceleration is needed (optional)
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# Install project
pip install -e .
```

## 🚀 Usage

### Basic Usage

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov12n.pt')  # or yolov12s/m/l/x.pt

# Enable TTA
model.enable_tta()

# Run inference
results = model.predict('path/to/image_or_video.mp4')
```

### Advanced Configuration

```python
# Load model
model = YOLO('yolov12n.pt')

# Enable TTA with custom parameters
model.enable_tta()
model.configure_tta(
    alpha=0.01,    # Learning rate - controls adaptation speed
    tau1=1.1,      # Primary threshold - detects major distribution shifts
    tau2=1.05,     # Secondary threshold - detects minor distribution shifts
    momentum=0.99  # EMA momentum - balances stability vs adaptability
)

# Video inference
results = model.predict('difficult_video.mp4')
```

### Parameter Tuning Recommendations

- **alpha**:

  - Higher values (0.05+) speed up adaptation but may be unstable
  - Lower values (0.001-) improve stability but adapt more slowly
  - Recommended range: 0.005-0.02

- **tau1/tau2**:
  - Higher values reduce adaptation trigger frequency, suitable for stable scenes
  - Lower values increase adaptation trigger frequency, better for rapidly changing scenes
  - Recommended combination: tau1=1.1, tau2=1.05

## 🖥️ Interactive Demo

```bash
# Start the web demo interface
python app.py

# Access in browser
# http://127.0.0.1:7860
```

The demo interface provides:

- Image and video input support
- TTA toggle and parameter adjustment
- Real-time detection preview
- With/without TTA comparison view

## 📊 Performance Benchmarking

```bash
# Run benchmark test
python benchmark.py --model yolov12n.pt --scenario domain_shift --visualize
```

Available test scenarios include:

- `standard`: Standard testing scenario
- `domain_shift`: Domain shift testing (e.g., day to night)
- `weather`: Weather condition changes
- `noise`: Noise robustness testing
- `blur`: Blur and motion blur testing

## Technical Details

TTA works through an online learning mechanism that allows the model to continuously adapt to new environments during inference:

1. Model identifies high-confidence predictions as "reliable samples"
2. These predictions are used as pseudo-labels for self-supervised learning
3. Dynamic thresholds determine when model updates are needed
4. Exponential moving average maintains model stability

This approach is particularly effective in scenarios with significant distribution shifts while maintaining performance in standard scenarios.

## Acknowledgements

- Original YOLOv12 provided by [sunsmarterjie](https://github.com/sunsmarterjie/yolov12)
- TTA implementation inspired by research in test-time adaptation and domain generalization

## License

This project is open-sourced under the AGPL-3.0 license.
