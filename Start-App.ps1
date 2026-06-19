# Start-App.ps1
# One-command launcher for the Large-Scale Animal Impound Handbook app (Windows)
# Usage: Right-click → Run with PowerShell, or:
#   powershell -ExecutionPolicy Bypass -File .\Start-App.ps1

$ErrorActionPreference = "Stop"

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host " Large-Scale Animal Impound Handbook - Streamlit App Launcher" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

$python = "C:\Users\just4\AppData\Local\Programs\Python\Python312\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "Python not found at expected location. Trying 'python' in PATH..." -ForegroundColor Yellow
    $python = "python"
}

# Go to script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "Working directory: $scriptDir" -ForegroundColor Gray

# Check for PDF
if (-not (Test-Path "document.pdf")) {
    Write-Error "document.pdf not found in current directory!"
    exit 1
}

# Install / upgrade requirements
Write-Host "`n[1/3] Ensuring Python packages are installed (this may take a few minutes on first run)..." -ForegroundColor Yellow
# Only (re)install if needed to avoid long/conflicting runs
if (-not (Test-Path "data/chroma_md")) {
    Write-Host "`n[1/3] Installing/updating packages (first run only)..." -ForegroundColor Yellow
    & $python -m pip install -r requirements.txt --quiet
    if ($LASTEXITCODE -ne 0) {
        & $python -m pip install -r requirements.txt
    }
} else {
    Write-Host "`n[1/3] Packages already installed (skipping full reinstall)." -ForegroundColor Green
}

# Optional: remind about Ollama
Write-Host "`n[2/3] Checking for Ollama (recommended for full Q&A)..." -ForegroundColor Yellow
try {
    $null = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 2 -ErrorAction Stop
    Write-Host "✅ Ollama is running." -ForegroundColor Green
} catch {
    Write-Host "🟡 Ollama not detected." -ForegroundColor Yellow
    Write-Host "   For the best natural language Q&A experience:" -ForegroundColor Gray
    Write-Host "   1. Install Ollama from https://ollama.com" -ForegroundColor Gray
    Write-Host "   2. Run:  ollama serve" -ForegroundColor Gray
    Write-Host "   3. In another terminal:  ollama pull llama3.2" -ForegroundColor Gray
    Write-Host "   The app will still work great with semantic search + retrieval only." -ForegroundColor Gray
}

# Launch
Write-Host "`n[3/3] Starting Streamlit app..." -ForegroundColor Green
Write-Host "The app will open in your browser shortly." -ForegroundColor Gray
Write-Host "Press Ctrl+C in this window to stop the app." -ForegroundColor DarkGray
Write-Host ""

& $python -m streamlit run app.py --server.headless true
