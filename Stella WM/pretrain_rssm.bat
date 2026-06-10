@echo off
echo ============================================================
echo   RSSM PRE-TRAINING - Stella World Model
echo   Python 3.12 + ROCm (GPU)
echo ============================================================
echo.

cd /d "D:\Stella WM"
set PY="C:\Users\Arcan\AppData\Local\Programs\Python\Python312\python.exe"

%PY% -m worldmodel.pretrain --epochs 15 --lr 3e-4 --wiki-articles 300

echo.
echo Listo.
pause
