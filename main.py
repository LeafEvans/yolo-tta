from ultralytics import YOLO

model = YOLO("yolo12n-tta-sense.pt")
for _ in model("sample.mp4", save=True, tta=True, tta_bn_only=False, stream=True):
    pass
