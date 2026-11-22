@echo off
rem Starts the CryptoHarm web UI. Close this window to stop it.
cd /d "%~dp0"
where py >nul 2>nul && (py -3 -m cryptoharm ui --open-browser) || (python -m cryptoharm ui --open-browser)
pause
