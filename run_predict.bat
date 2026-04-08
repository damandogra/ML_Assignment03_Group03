@echo off
setlocal

REM ===== SETTINGS =====
set MODEL=runs/detect/wieger_detect_s/weights/best.pt
set SOURCE=dataset\validate\images
set DEVICE=0
set CONF=0.05
set WORKERS=0
set RUNNAME=wieger_pred

REM ===== PREDICT =====
yolo detect predict ^
model=%MODEL% ^
source=%SOURCE% ^
device=%DEVICE% ^
conf=%CONF% ^
workers=%WORKERS% ^
save_txt ^
save_conf ^
name=%RUNNAME%

pause