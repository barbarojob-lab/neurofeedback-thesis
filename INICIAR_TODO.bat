@echo off
setlocal

cd /d "%~dp0"
set "ROOT=%CD%"
set "PYTHON_EXE=%ROOT%\.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
  echo [ERROR] No se encontro el entorno virtual en .venv\Scripts\python.exe
  echo         Crea/activa el venv antes de iniciar todo.
  pause
  exit /b 1
)

echo Iniciando servicios en ventanas separadas...

start "ML Service :8001" powershell -NoExit -ExecutionPolicy Bypass -Command "Set-Location '%ROOT%\ml-service'; & '%PYTHON_EXE%' server.py"
start "Backend WS :8080" powershell -NoExit -ExecutionPolicy Bypass -Command "Set-Location '%ROOT%\backend'; npm run dev"
start "Frontend Vite :5173" powershell -NoExit -ExecutionPolicy Bypass -Command "Set-Location '%ROOT%\frontend'; npm run dev"

timeout /t 2 /nobreak > nul
start "" "http://localhost:5173"

echo Listo. Si algun servicio falla, revisa su ventana.
endlocal