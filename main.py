from ultralytics import YOLO

model = YOLO("yolo12n-tta-sense.pt")

for _ in model("sample.mp4", tta=False, tta_update_mode="adaptor", tta_loss_mode="unweighted_obj", stream=True):
    pass
