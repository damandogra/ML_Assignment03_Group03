@echo off
setlocal

REM ===== TUNABLE SETTINGS =====
set MODEL=yolo26n-cls.pt
set DATA=.
set EPOCHS=100
set IMGSZ=224
set BATCH=32
set DEVICE=0
set WORKERS=0
set AMP=False
set PATIENCE=10
set RUNNAME=cls26_full

REM ===== TRAIN =====
echo ========================================
echo STARTING BINARY CLASSIFIER TRAINING
echo MODEL=%MODEL%
echo DATA=%DATA%
echo EPOCHS=%EPOCHS%
echo IMGSZ=%IMGSZ%
echo BATCH=%BATCH%
echo DEVICE=%DEVICE%
echo WORKERS=%WORKERS%
echo AMP=%AMP%
echo PATIENCE=%PATIENCE%
echo RUNNAME=%RUNNAME%
echo ========================================

yolo classify train ^
model=%MODEL% ^
data=%DATA% ^
epochs=%EPOCHS% ^
imgsz=%IMGSZ% ^
batch=%BATCH% ^
device=%DEVICE% ^
workers=%WORKERS% ^
amp=%AMP% ^
patience=%PATIENCE% ^
name=%RUNNAME%

pause