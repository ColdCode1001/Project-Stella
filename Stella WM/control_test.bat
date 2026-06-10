@echo off
echo ============================================================
echo   TEST DE CONTROL — el gate de STELLA_FUNDAMENTO §3
echo   Prueba si el World Model controla la boca (sin entrenar)
echo ============================================================
echo.

cd /d "D:\Stella WM"
set PY="C:\Users\Arcan\AppData\Local\Programs\Python\Python312\python.exe"

%PY% -m worldmodel.control_test --k 6 --decoder stella

echo.
pause
