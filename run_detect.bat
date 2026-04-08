@echo off
setlocal

REM ===== SETTINGS =====
set MODEL=yolov8s.pt
set DATA=data.yaml
set EPOCHS=30
set IMGSZ=640
set BATCH=4
set DEVICE=0
set WORKERS=0
set AMP=False
set CACHE=False
set RUNNAME=wieger_detect_s

REM ===== TRAIN =====
echo ========================================
echo STARTING TRAINING
echo ========================================
yolo detect train ^
model=%MODEL% ^
data=%DATA% ^
epochs=%EPOCHS% ^
imgsz=%IMGSZ% ^
batch=%BATCH% ^
device=%DEVICE% ^
workers=%WORKERS% ^
amp=%AMP% ^
cache=%CACHE% ^
name=%RUNNAME%

pause