param(
    [string]$PythonExe = ".\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

Write-Host "Using Python: $PythonExe"

& $PythonExe -m PyInstaller `
    --noconfirm `
    --clean `
    --onedir `
    --name "HVAC_Cashflow" `
    --collect-all streamlit `
    --collect-all plotly `
    --collect-all altair `
    --collect-all reportlab `
    --collect-all kaleido `
    --collect-all openpyxl `
    --collect-all xlsxwriter `
    --add-data "app.py;." `
    --add-data "src;src" `
    launcher.py

Write-Host ""
Write-Host "Build complete."
Write-Host "Executable: dist\HVAC_Cashflow\HVAC_Cashflow.exe"
Write-Host "Share the full dist\HVAC_Cashflow folder (zip it for easiest distribution)."
