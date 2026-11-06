# a4ai Agent on REAL a4ai data, READ-ONLY (Windows PowerShell). Run after agent/sql/create_agent_ro_role.sql.
#   powershell -ExecutionPolicy Bypass -File scripts\run_agent_real.ps1 -ProjectRef abcdefghijklmnop -Region ap-south-1
# Asks for the agent_ro password without echoing it and never writes it to disk or history.
# Needs A4AI_ANTHROPIC_API_KEY (and optionally GEMINI_API_KEY) already set in this shell.
param(
  [Parameter(Mandatory = $true)][string]$ProjectRef,
  [string]$Region = "ap-south-1",
  [int]$Port = 8000,
  [string]$CorsOrigins = "http://localhost:5173,http://localhost:8080"
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = Join-Path $Root "agent\.devdata\venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { throw "Run scripts\run_agent_dev.ps1 once first (it creates the venv and installs deps)" }

# real and dev mode are mutually exclusive; the app also refuses to start if both are set
Remove-Item Env:AGENT_DEV_MODE, Env:AGENT_DEV_AUTH, Env:AGENT_DB -ErrorAction SilentlyContinue

$sec = Read-Host -AsSecureString "agent_ro password (from create_agent_ro_role.sql)"
$pw = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec))
$enc = [Uri]::EscapeDataString($pw)
# Session pooler = IPv4 (the direct db.<ref>.supabase.co host is IPv6-only on many plans). User = agent_ro.<project-ref>
$env:AGENT_REAL_DB_URL = "postgresql://agent_ro.${ProjectRef}:${enc}@aws-0-${Region}.pooler.supabase.com:5432/postgres?sslmode=require"
$env:AGENT_REAL_DB_MODE = "1"
$env:SUPABASE_URL = "https://${ProjectRef}.supabase.co"
$env:AGENT_CORS_ORIGINS = $CorsOrigins
if (-not $env:SUPABASE_JWT_SECRET) {
  Write-Host "SUPABASE_JWT_SECRET not set: tokens are verified with the project's JWKS (asymmetric signing keys)."
  Write-Host "If your project still uses the legacy JWT secret, set `$env:SUPABASE_JWT_SECRET first."
}
foreach ($k in "A4AI_ANTHROPIC_API_KEY") {
  if (-not [Environment]::GetEnvironmentVariable($k)) { Write-Warning "$k is not set" }
}
Write-Host "Starting a4ai Agent (LIVE, read-only) on http://localhost:$Port"
& $Py -m uvicorn agent.api:app --host 127.0.0.1 --port $Port
$code = $LASTEXITCODE
Remove-Item Env:AGENT_REAL_DB_URL -ErrorAction SilentlyContinue
exit $code
