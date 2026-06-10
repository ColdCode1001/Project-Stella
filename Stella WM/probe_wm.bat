@echo off
echo ============================================================
echo   PROBE — Inspector del World Model (sin decoder)
echo ============================================================
echo.

cd /d "D:\Stella WM"
set PY="C:\Users\Arcan\AppData\Local\Programs\Python\Python312\python.exe"

if "%1"=="large" (
    echo Usando LargeRSSM (128.8M params)
    %PY% -m worldmodel.probe_wm --model large
) else (
    echo Usando MinimalRSSM (550K params)
    echo Para el Large: probe_wm.bat large
    %PY% -m worldmodel.probe_wm
)

pause
