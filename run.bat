@echo off
cd /d "%~dp0"

start "Ollama Server" /min ollama serve
ping -n 4 127.0.0.1 >nul

python main.py

pause
