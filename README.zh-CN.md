# YOLOv12-TTA 测试时适配

<div align="center">
<p>
  <a href="README.md">English</a> | <b>简体中文</b>
</p>

<h2>YOLOv12: 具有测试时适配能力的注意力中心实时目标检测器</h2>

作者: <a href="mailto:leafevans@foxmail.com">LeafEvans</a>

基于 YOLOv12 原始工作的改进实现

</div>

## 🔥 项目简介

YOLOv12-TTA 为 YOLOv12 添加了**测试时适配**能力，使模型能够在不重新训练的情况下实时适应变化的环境。这项技术特别适用于以下场景：

- 环境条件变化（如白天/夜晚、晴天/雨天）
- 相机角度和位置变化
- 光照条件剧烈波动
- 需要在未见过的环境中保持检测精度

## ⚡ TTA 核心优势

- **持续学习**：模型在推理过程中不断适应新环境
- **无需重训**：避免了耗时的域适应训练过程
- **动态调节**：根据检测置信度自动判断何时需要适配
- **轻量级**：额外计算开销小于 5%，几乎不影响实时性

## 📊 性能表现

| 场景                 | 标准 YOLOv12n | 启用 TTA | 提升幅度   |
| -------------------- | ------------- | -------- | ---------- |
| 标准条件             | 40.4 mAP      | 40.7 mAP | +0.7%      |
| 域偏移 (白天 → 夜晚) | 28.3 mAP      | 35.6 mAP | **+25.8%** |
| 天气变化 (雾天)      | 30.1 mAP      | 36.2 mAP | **+20.3%** |
| 添加噪声             | 32.5 mAP      | 37.3 mAP | **+14.8%** |
| 运动模糊             | 31.7 mAP      | 36.8 mAP | **+16.1%** |

\*测试基于 COCO 验证集图像并应用合成变换

## 📦 安装指南

```bash
# 安装依赖
pip install -r requirements.txt

# 如果需要 flash-attention 加速（可选）
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# 安装项目
pip install -e .
```

## 🚀 使用方法

### 基本用法

```python
from ultralytics import YOLO

# 加载模型
model = YOLO('yolov12n.pt')  # 或选择 yolov12s/m/l/x.pt

# 启用TTA
model.enable_tta()

# 进行推理
results = model.predict('path/to/image_or_video.mp4')
```

### 高级配置

```python
# 加载模型
model = YOLO('yolov12n.pt')

# 启用TTA并自定义参数
model.enable_tta()
model.configure_tta(
    alpha=0.01,    # 学习率 - 控制适应速度
    tau1=1.1,      # 主阈值 - 用于检测主要分布偏移
    tau2=1.05,     # 次阈值 - 用于检测轻微分布偏移
    momentum=0.99  # EMA动量 - 平衡稳定性与适应性
)

# 视频推理
results = model.predict('difficult_video.mp4')
```

### 参数调优建议

- **alpha**:

  - 增大值(0.05+)加速适应但可能不稳定
  - 减小值(0.001-)提高稳定性但适应较慢
  - 推荐范围: 0.005-0.02

- **tau1/tau2**:
  - 增大值减少适应触发频率，适合稳定场景
  - 减小值增加适应触发频率，适合快速变化场景
  - 推荐组合: tau1=1.1, tau2=1.05

## 🖥️ 交互式演示

```bash
# 启动Web演示界面
python app.py

# 在浏览器中访问
# http://127.0.0.1:7860
```

演示界面提供了以下功能：

- 图像和视频输入支持
- TTA 开关和参数调整
- 实时检测结果预览
- 有/无 TTA 对比视图

## 📊 性能基准测试

```bash
# 运行基准测试
python benchmark.py --model yolov12n.pt --scenario domain_shift --visualize
```

可测试的场景包括：

- `standard`: 标准测试场景
- `domain_shift`: 域偏移测试(如白天到夜晚)
- `weather`: 天气变化测试
- `noise`: 噪声鲁棒性测试
- `blur`: 模糊和运动模糊测试

## 技术细节

TTA 的工作原理是通过在线学习机制，让模型在推理过程中持续适应新环境：

1. 模型识别高置信度预测作为"可靠样本"
2. 将这些预测作为伪标签进行自监督学习
3. 使用动态阈值判断何时需要更新模型
4. 通过指数移动平均保持模型稳定性

该方法在分布偏移较大的场景中尤其有效，同时在标准场景中不会降低性能。

## 致谢

- 原始 YOLOv12 由 [sunsmarterjie](https://github.com/sunsmarterjie/yolov12) 提供
- TTA 实现受到测试时适配和域泛化研究的启发

## 许可证

本项目基于 AGPL-3.0 许可证开源。
