@echo off
setlocal

set MODEL=yolov8s.pt
set DATA=data.yaml

yolo detect train ^
model=%MODEL% ^
data=%DATA% ^
epochs=1 ^
imgsz=640 ^
batch=4 ^
device=0 ^
workers=0 ^
amp=False ^
cache=False ^
name=wieger_quickcheck

pause