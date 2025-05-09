from ultralytics import YOLO

model = YOLO("yolo12n-tta.pt")

model.val(data="coco.yaml",tta=True)
