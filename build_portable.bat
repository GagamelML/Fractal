@echo off
echo ============================================
echo  Building Fractal Studio portable package
echo ============================================
echo.

:: Activate venv if present
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

echo [1/2] Running PyInstaller...
pyinstaller fractal_studio.spec --noconfirm

echo.
echo [2/2] Copying results folder structure...
if not exist dist\FractalStudio\results mkdir dist\FractalStudio\results

echo.
echo ============================================
echo  Done!  Distributable folder:
echo    dist\FractalStudio\
echo.
echo  Run:  dist\FractalStudio\FractalStudio.exe
echo ============================================
pause

