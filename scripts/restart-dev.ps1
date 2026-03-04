$ErrorActionPreference = "Stop"

function Stop-NextRelatedProcesses {
  # Best-effort: kill any "next dev" / "npm run dev" processes that might not be listening
  try {
    $procs = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
      $_.CommandLine -and (
        $_.CommandLine -match "next(\.exe)?\s+dev" -or
        $_.CommandLine -match "npm(\.cmd)?\s+run\s+dev" -or
        $_.CommandLine -match "node(\.exe)?\s+.*next"
      )
    }
    foreach ($p in ($procs | Select-Object -Unique ProcessId)) {
      try { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue } catch {}
    }
  } catch {
    # ignore
  }
}

function Stop-ListeningPort {
  param([Parameter(Mandatory = $true)][int]$Port)
  try {
    $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    foreach ($c in ($conns | Select-Object -Unique OwningProcess)) {
      if ($null -ne $c.OwningProcess -and $c.OwningProcess -gt 0) {
        try { Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue } catch {}
      }
    }
  } catch {
    # ignore if Get-NetTCPConnection isn't available
  }
}

$root = Split-Path -Parent $PSScriptRoot
$backendDir = Join-Path $root "backend"
$webDir = Join-Path $root "web"
$venvPy = Join-Path $backendDir ".venv\Scripts\python.exe"

Write-Host "Stopping existing dev servers (ports 8000, 3000, 3001)..." -ForegroundColor Cyan
Stop-NextRelatedProcesses
Stop-ListeningPort -Port 8000
Stop-ListeningPort -Port 3000
Stop-ListeningPort -Port 3001

# Clear Next dev lock if it exists
$nextLock = Join-Path $webDir ".next\dev\lock"
if (Test-Path $nextLock) {
  Write-Host "Removing Next lock file..." -ForegroundColor Cyan
  try { Remove-Item -Force $nextLock } catch {}
}

Write-Host "Starting backend (FastAPI)..." -ForegroundColor Green
$backendPy = "python"
if (Test-Path $venvPy) {
  $backendPy = $venvPy
}
Start-Process -WorkingDirectory $backendDir -FilePath "powershell.exe" -ArgumentList @(
  "-NoExit",
  "-Command",
  "& `"$backendPy`" -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
)

Write-Host "Starting frontend (Next.js)..." -ForegroundColor Green
Start-Process -WorkingDirectory $webDir -FilePath "powershell.exe" -ArgumentList @(
  "-NoExit",
  "-Command",
  "npm run dev -- --port 3000"
)

Start-Sleep -Seconds 2
try {
  $listen = Get-NetTCPConnection -State Listen -LocalPort 3000,3001 -ErrorAction SilentlyContinue | Select-Object -First 1 LocalPort
  if ($listen -and $listen.LocalPort) {
    Start-Process "http://localhost:$($listen.LocalPort)/"
  }
} catch {
  # ignore
}

Write-Host ""
Write-Host "If a port is still busy, just re-run this script." -ForegroundColor DarkGray
