@echo off
rem Opens the Person Extractor window (gui.py) with the project's own Python.
cd /d "%~dp0"
start "" ".venv\Scripts\pythonw.exe" gui.py
