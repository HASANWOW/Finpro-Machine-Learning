@echo off
echo ===========================================
echo Setup Project FoodVibe
echo ===========================================

echo.
echo [1/3] Membuat Virtual Environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo Gagal membuat virtual environment! Pastikan Python sudah terinstall dan masuk PATH.
    pause
    exit /b %errorlevel%
)

echo.
echo [2/3] Mengaktifkan Virtual Environment dan Update PIP...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip

echo.
echo [3/3] Menginstall Dependencies...
pip install -r requirements.txt

echo.
echo ===========================================
echo Setup Selesai!
echo Untuk mulai mengaktifkan environment, jalankan:
echo venv\Scripts\activate
echo ===========================================
pause
