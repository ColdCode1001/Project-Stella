@echo off
chcp 65001 >nul
title Stella WM Demo
cd /d "D:\Stella WM"
set PYTHONIOENCODING=utf-8

echo.
echo  STELLA WORLD MODEL DEMO
echo  ========================
echo  Iniciando servidor...
echo.

:: Matar instancia previa si existe
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":5001 "') do (
    taskkill /PID %%a /F >nul 2>&1
)

set PYTHON=C:\Users\Arcan\AppData\Local\Programs\Python\Python311\python.exe

:: Arrancar el servidor en background
start "StellaWM-Server" /min cmd /c "chcp 65001 >nul && set PYTHONIOENCODING=utf-8 && %PYTHON% demo_dashboard.py > logs\demo.log 2>&1"

:: Esperar a que el servidor levante
echo  Esperando que el servidor levante...
timeout /t 3 /nobreak >nul

:wait_loop
%PYTHON% -c "import urllib.request; urllib.request.urlopen('http://localhost:5001')" >nul 2>&1
if errorlevel 1 (
    timeout /t 1 /nobreak >nul
    goto wait_loop
)

echo  Servidor listo.
echo.

:: Abrir el navegador
start "" "http://localhost:5001"

echo  Dashboard abierto en http://localhost:5001
echo  Cierra esta ventana para detener el servidor.
echo.
pause

:: Al cerrar, matar el servidor
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":5001 "') do (
    taskkill /PID %%a /F >nul 2>&1
)
