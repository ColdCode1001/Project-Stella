@echo off
chcp 65001 >nul
title EXP-001 -- SmallDecoder Training
cd /d "D:\Stella WM"
set PYTHONIOENCODING=utf-8

set PYTHON=C:\Users\Arcan\AppData\Local\Programs\Python\Python312\python.exe

echo.
echo  EXP-001 -- SmallDecoder Training
echo  ==================================
echo  GPU: RX 7900 XTX (ROCm)
echo  Decoder: 38.6M params desde cero
echo  Datos: DailyDialog + chats de Stella
echo.

mkdir logs 2>nul
%PYTHON% -m worldmodel.train_decoder --epochs 5 --batch-size 32 --lr 1e-4 2>&1 > logs\train_decoder.log

echo.
echo  Entrenamiento completado.
echo  Revisa logs\train_decoder.log para ver los resultados.
pause
