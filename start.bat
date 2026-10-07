@echo off
REM ===========================================================================
REM  start.bat - launch ET News Intelligence
REM
REM  Double-click this file, or run  start.bat  from a terminal.
REM
REM  What it does, in order:
REM    1. moves to the project folder
REM    2. checks the virtual environment exists
REM    3. starts Ollama if it is not already running
REM    4. warns if a required model is missing
REM    5. checks the dataset and the embedding matrix are present
REM    6. opens the Streamlit app in your browser
REM
REM  Note: it deliberately calls .venv\Scripts\python.exe instead of running
REM  Activate.ps1, because PowerShell often blocks script execution by policy.
REM ===========================================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

set "PYTHON=%~dp0.venv\Scripts\python.exe"
set "PORT=8501"
set "EMBED_MODEL=qwen3-embedding:0.6b"
set "LLM_MODEL=mistral:latest"
set "OLLAMA=ollama"

echo.
echo  ==========================================================
echo   ET NEWS INTELLIGENCE
echo  ==========================================================
echo.

REM --- 1. virtual environment ------------------------------------------------
if not exist "%PYTHON%" (
    echo  [X] Virtual environment not found at .venv
    echo.
    echo      Create it first:
    echo        python -m venv .venv
    echo        .venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)
echo  [ok] virtual environment found

REM --- 2. Python dependencies ------------------------------------------------
"%PYTHON%" -c "import streamlit, pandas, numpy, ollama" >nul 2>&1
if errorlevel 1 (
    echo  [X] Some Python packages are missing. Installing from requirements.txt...
    "%PYTHON%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo  [X] Install failed. Check your internet connection and try again.
        pause
        exit /b 1
    )
)
echo  [ok] python packages ready

REM --- 3. Ollama -------------------------------------------------------------
"%OLLAMA%" list >nul 2>&1
if errorlevel 1 (
    echo  [*] Ollama is not running, starting it now...
    start "Ollama" /min "%OLLAMA%" serve
    timeout /t 6 /nobreak >nul
    "%OLLAMA%" list >nul 2>&1
    if errorlevel 1 (
        echo  [X] Could not start Ollama.
        echo      Please run "ollama serve" in another terminal, then start this again.
        pause
        exit /b 1
    )
)
echo  [ok] Ollama is running

REM --- 4. Models -------------------------------------------------------------
"%OLLAMA%" list | findstr /i "qwen3-embedding" >nul
if errorlevel 1 (
    echo  [!] Embedding model %EMBED_MODEL% is not pulled. Downloading now...
    "%OLLAMA%" pull %EMBED_MODEL%
)
"%OLLAMA%" list | findstr /i "mistral" >nul
if errorlevel 1 (
    echo  [!] Language model %LLM_MODEL% is not pulled. Downloading now...
    echo      This is about 4 GB and can take a while.
    "%OLLAMA%" pull %LLM_MODEL%
)
echo  [ok] models ready

REM --- 5. Dataset and embeddings --------------------------------------------
if not exist "Data\articles_clean.csv" (
    echo  [!] Data\articles_clean.csv is missing. Building the dataset...
    "%PYTHON%" src\combine_data.py
    "%PYTHON%" src\clean_data.py
)
if not exist "models\article_embeddings.npy" (
    echo  [!] models\article_embeddings.npy is missing. Building embeddings...
    echo      This takes about 10 minutes on a laptop GPU. Please wait.
    "%PYTHON%" src\create_embeddings.py
)
echo  [ok] dataset and embeddings ready

REM --- 6. Launch -------------------------------------------------------------
echo.
echo  --------------------------------------------------------------
echo   Starting the app on http://localhost:%PORT%
echo   Close this window, or press Ctrl+C, to stop it.
echo  --------------------------------------------------------------
echo.

start "" "http://localhost:%PORT%"
"%PYTHON%" -m streamlit run app\app.py --server.port %PORT% --browser.gatherUsageStats false

echo.
echo  The app has stopped.
pause
endlocal