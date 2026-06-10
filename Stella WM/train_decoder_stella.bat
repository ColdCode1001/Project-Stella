@echo off
echo ============================================================
echo   DECODER - Voz de Stella
echo   Fine-tune en chats + thoughts + episodes + web triggers
echo   Python 3.12 + ROCm (GPU)
echo ============================================================
echo.

cd /d "D:\Stella WM"
set PY="C:\Users\Arcan\AppData\Local\Programs\Python\Python312\python.exe"

%PY% -m worldmodel.train_decoder_stella --epochs 100 --batch-size 8 --lr 2e-5

echo.
echo Listo. Para activar: copia decoder_stella.pt a decoder.pt
pause
