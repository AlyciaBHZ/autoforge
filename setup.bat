@echo off
setlocal enabledelayedexpansion

echo === AutoForge Setup (Windows) ===
echo.

:: ── Check Python ──
set "PYTHON_CMD="
for %%P in (python3 python) do (
    where %%P >nul 2>&1
    if !errorlevel! equ 0 (
        %%P -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" 2>nul
        if !errorlevel! equ 0 (
            set "PYTHON_CMD=%%P"
            goto :found_python
        )
    )
)

echo Error: Python 3.11+ required
echo Install from: https://python.org
echo Make sure to check "Add Python to PATH" during installation
exit /b 1

:found_python
for /f "tokens=*" %%V in ('%PYTHON_CMD% --version') do echo Python: %%V

:: ── Create virtual environment ──
if not exist ".venv" (
    echo Creating virtual environment...
    %PYTHON_CMD% -m venv .venv
)
echo Virtual environment: .venv

:: ── Activate and install dependencies ──
call .venv\Scripts\activate.bat
echo Installing dependencies...
pip install -q --upgrade pip
pip install -q -r requirements.txt

:: ── Create .env ──
if not exist ".env" (
    copy .env.example .env >nul
    echo.
    echo Created .env from template.
    echo ^>^>^> Please edit .env and add your ANTHROPIC_API_KEY ^<^<^<
    echo.
)

:: ── Create workspace ──
if not exist "workspace" mkdir workspace

:: ── Check Docker (optional) ──
echo.
where docker >nul 2>&1
if %errorlevel% equ 0 (
    echo Docker: found
    docker info >nul 2>&1
    if !errorlevel! equ 0 (
        echo Docker is running
        if exist "docker\Dockerfile.sandbox" (
            echo Building sandbox image...
            docker build -t autoforge-sandbox:latest -f docker\Dockerfile.sandbox docker\ >nul 2>&1
            if !errorlevel! equ 0 (
                echo Sandbox image built successfully
                findstr /c:"FORGE_DOCKER_ENABLED" .env >nul 2>&1
                if !errorlevel! neq 0 (
                    echo FORGE_DOCKER_ENABLED=true>> .env
                )
            ) else (
                echo Warning: Docker build failed. Sandbox will use subprocess mode.
            )
        )
    ) else (
        echo Warning: Docker found but not running. Sandbox will use subprocess mode.
    )
) else (
    echo Docker: not found ^(optional^). Sandbox will use subprocess mode.
)

:: ── Verify installation ──
echo.
echo Verifying installation...
.venv\Scripts\python -c "from engine.orchestrator import Orchestrator; print('  Engine: OK')" 2>&1 || echo   Engine: FAILED

echo.
echo === Setup complete! ===
echo.
echo Next steps:
echo   1. Edit .env and add your ANTHROPIC_API_KEY
echo   2. Run: .venv\Scripts\activate.bat
echo   3. Run: python forge.py "your project description"
echo.
echo Examples:
echo   python forge.py "Build a Todo app with user login"
echo   python forge.py "做一个心理咨询预约平台" --budget 5.00

endlocal
