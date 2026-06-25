@echo off
echo ================================================
echo  ContractIQ Backend — Starting up
echo ================================================

cd /d "%~dp0"

REM Install dependencies if not already installed
pip install -r requirements.txt --quiet

echo.
echo Starting FastAPI server on http://localhost:8000
echo API docs available at http://localhost:8000/docs
echo.
echo NOTE: First startup builds the CUAD index (~30-90s depending on hardware)
echo.

uvicorn main:app --reload --port 8000 --log-level info
