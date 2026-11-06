# a4ai Agent - local dev on plain Windows PowerShell. No Docker, no Git Bash, no Postgres install:
# the database is embedded Postgres from the `pgserver` pip wheel (Python 3.9-3.12; no 3.13 wheel yet).
#   powershell -ExecutionPolicy Bypass -File scripts\run_agent_dev.ps1 [-Port 8000] [-KeepData] [-WithBackendRequirements]
# Reads A4AI_ANTHROPIC_API_KEY / GEMINI_API_KEY from the environment (never printed). Refuses to start if a
# production DB URL (Supabase etc.) is configured in this shell.
param(
  [int]$Port = 8000,
  [switch]$KeepData,
  [switch]$WithBackendRequirements
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Venv = Join-Path $Root "agent\.devdata\venv"   # gitignored (agent/.gitignore)
$Py = Join-Path $Venv "Scripts\python.exe"

# Native commands that write to stderr must not trip $ErrorActionPreference = "Stop" (PowerShell 5.1),
# so they run under "Continue" and are judged by $LASTEXITCODE only.
function Invoke-Native([string]$Exe, [string[]]$NativeArgs) {
  $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
  try { & $Exe @NativeArgs 2>&1 | ForEach-Object { "$_" } | Out-Host } finally { $ErrorActionPreference = $old }
  return $LASTEXITCODE
}

$UseUv = $false
if (-not (Test-Path $Py)) {
  Write-Host "Creating Python 3.12 venv in agent\.devdata\venv"
  $made = $false
  if (Get-Command py -ErrorAction SilentlyContinue) {
    $made = ((Invoke-Native "py" @("-3.12", "-m", "venv", $Venv)) -eq 0) -and (Test-Path $Py)
  }
  if (-not $made) {
    # No Python 3.12 installed: uv (installed with pip) downloads a standalone 3.12 for this venv only.
    Write-Host "Python 3.12 not found via 'py'; using uv to fetch one (pip install --user uv)"
    if ((Invoke-Native "python" @("-m", "pip", "install", "--user", "--quiet", "uv")) -ne 0) { throw "pip install uv failed" }
    if ((Invoke-Native "python" @("-m", "uv", "venv", "--python", "3.12", $Venv)) -ne 0 -or -not (Test-Path $Py)) {
      throw "Could not create a Python 3.12 venv (pgserver has no Python 3.13 wheel)"
    }
  }
}
# uv-made venvs have no pip; use uv for installs there
if ((Invoke-Native $Py @("-m", "pip", "--version")) -ne 0) { $UseUv = $true }

function Install-Req([string]$File) {
  if ($UseUv) { return Invoke-Native "python" @("-m", "uv", "pip", "install", "--python", $Py, "-q", "-r", $File) }
  return Invoke-Native $Py @("-m", "pip", "install", "-q", "-r", $File)
}

Write-Host "Installing agent dependencies"
if ((Install-Req "agent\requirements-dev.txt") -ne 0) { throw "pip install failed" }
if ($WithBackendRequirements) {
  if ((Install-Req "test-generator-backend\requirements.txt") -ne 0) { Write-Warning "Backend requirements failed; the agent itself still runs" }
}

$env:AGENT_DEV_MODE = "1"
$env:AGENT_DB = "embedded"
if (-not $env:AGENT_CORS_ORIGINS) { $env:AGENT_CORS_ORIGINS = "http://localhost:5173,http://localhost:8080" }
foreach ($k in "A4AI_ANTHROPIC_API_KEY", "GEMINI_API_KEY") {
  if (-not [Environment]::GetEnvironmentVariable($k)) { Write-Warning "$k is not set" }
}

$args2 = @("-m", "agent.dev_server", "--port", "$Port")
if ($KeepData) { $args2 += "--keep-data" }
& $Py @args2
exit $LASTEXITCODE
