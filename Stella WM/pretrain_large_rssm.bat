@echo off
echo ============================================================
echo   PRETRAIN LargeRSSM (128.8M params)
echo   Wikipedia ES + datos de Stella
echo   Python 3.12 + ROCm (GPU encode, CPU train)
echo ============================================================
echo.

cd /d "D:\Stella WM"
set PY="C:\Users\Arcan\AppData\Local\Programs\Python\Python312\python.exe"

%PY% -m worldmodel.pretrain_large --epochs 10 --max-articles 2000 --max-seqs 50000 --lr 1e-4

echo.
echo Listo. rssm_large.pt guardado en worldmodel/weights/
echo Para probar: probe_wm.bat large
pause
