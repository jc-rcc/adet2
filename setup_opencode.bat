@echo off
setlocal

echo =============================================
echo  OpenCode Setup
echo =============================================

where opencode >nul 2>nul
if %errorlevel%==0 (
    echo [OK] opencode is already installed
    goto run
)

echo [..] opencode not found - checking Node.js...

where node >nul 2>nul
if %errorlevel%==0 (
    echo [OK] Node.js found
    goto install
)

echo [..] Node.js not found - installing Node.js via winget...
winget install --id OpenJS.NodeJS.LTS --exact --silent --accept-package-agreements --accept-source-agreements
if %errorlevel% neq 0 (
    echo [FAIL] Failed to install Node.js. Please install Node.js manually from https://nodejs.org
    pause
    exit /b 1
)

:install
echo [..] Installing opencode via npm...
call npm install -g opencode-ai
if %errorlevel% neq 0 (
    echo [FAIL] Failed to install opencode.
    pause
    exit /b 1
)

echo [OK] opencode installed successfully.

:run
echo =============================================
echo  Starting opencode...
echo =============================================
opencode