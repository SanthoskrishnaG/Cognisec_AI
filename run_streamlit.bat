@echo off
cd /d "%~dp0"
echo ================================================================================
echo   Cognisec AI: Launching Legacy Streamlit Demonstration Dashboard
echo ================================================================================
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m streamlit run app.py %*
) else (
    echo [!] COGNISEC AI: Virtual environment not found (.venv\Scripts\python.exe).
    echo Please create the virtual environment using Python 3.12:
    echo   py -3.12 -m venv .venv
    echo   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)
pause
