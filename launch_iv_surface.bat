@echo off
echo Checking for virtual environment...

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate

echo Checking for required packages...

REM Check if each package is installed, and install if not
pip show streamlit >nul 2>&1 || (
    echo Installing streamlit...
    pip install streamlit
)

pip show yfinance >nul 2>&1 || (
    echo Installing yfinance...
    pip install yfinance
)

pip show mibian >nul 2>&1 || (
    echo Installing mibian...
    pip install mibian
)

pip show plotly >nul 2>&1 || (
    echo Installing plotly...
    pip install plotly
)

pip show scipy >nul 2>&1 || (
    echo Installing scipy...
    pip install scipy
)

echo Launching IV Surface Explorer...
call venv\Scripts\streamlit run "%~dp0iv_surface_app.py"
pause
