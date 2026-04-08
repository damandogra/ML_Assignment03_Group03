@echo off

echo === TRAIN (GPU SAFE) ===
yolo detect train model=yolov8n.pt data=data.yaml epochs=10 imgsz=512 batch=2 device=0 workers=0 amp=False cache=False name=myrun

echo === VALIDATE ===
yolo detect val model=runs/detect/myrun/weights/best.pt data=data.yaml device=0 workers=0

echo === PREDICT TEST ===
yolo detect predict model=runs/detect/myrun/weights/best.pt source=PATH_TO_TEST_IMAGES device=0 save_txt save_conf workers=0

pause