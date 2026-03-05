@echo off
setlocal enabledelayedexpansion

echo === AutoForge Setup (Windows) ===
echo.

:: ── Check Python ──
set "PYTHON_CMD="
for %%P in (python3 python) do (
    where %%P >nul 2>&1
    if !errorlevel! equ 0 (
        %%P -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" 2>nul
        if !errorlevel! equ 0 (
            set "PYTHON_CMD=%%P"
            goto :found_python
        )
    )
)

echo Error: Python 3.10+ required
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

:: ── Activate and install ──
call .venv\Scripts\activate.bat
echo Installing ForgeAI...
pip install -q --upgrade pip
pip install -q -e .

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
.venv\Scripts\forgeai --help >nul 2>&1 && echo   ForgeAI CLI: OK || echo   ForgeAI CLI: FAILED
.venv\Scripts\python -c "from autoforge.engine.orchestrator import Orchestrator; print('  Engine: OK')" 2>&1 || echo   Engine: FAILED

:: ── Auto-launch setup wizard if needed ──
echo.
.venv\Scripts\python -c "from autoforge.cli.setup_wizard import needs_setup; exit(0 if needs_setup() else 1)" 2>nul
if !errorlevel! equ 0 (
    echo First-time setup — launching configuration wizard...
    echo.
    .venv\Scripts\forgeai setup
)

echo.
echo === Setup complete! ===
echo.
echo Usage:
echo   forgeai                                      :: Interactive session
echo   forgeai generate "your prompt"               :: Generate a project
echo   forgeai review .\my-project                  :: Review existing code
echo   forgeai setup                                :: Reconfigure settings
echo.
echo Or install globally:
echo   pip install -e .    :: Then "forgeai" works from anywhere
echo.

endlocal
