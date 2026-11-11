# Shared by run_agent_dev.ps1 / run_agent_real.ps1 (dot-source it). Local dev convenience only:
#   1. keys that are not in this shell are read from <repo>\.env (whitelist below; values are never printed)
#   2. no Anthropic key but a Gemini key -> AGENT_PROVIDER=gemini (unless AGENT_PROVIDER is already set)
#   3. -Voice -> VOICE_ENABLED=1 (the mic in the UI appears only when this is on and SARVAM_API_KEY is set)
# Production never runs this file: it sets its own environment.
param([switch]$Voice)

$Root = Split-Path -Parent $PSScriptRoot
$Allowed = "A4AI_ANTHROPIC_API_KEY", "GEMINI_API_KEY", "SARVAM_API_KEY", "SUPABASE_JWT_SECRET"
$DotEnv = Join-Path $Root ".env"
if (Test-Path $DotEnv) {
  foreach ($line in Get-Content $DotEnv) {
    $t = $line.Trim()
    if ($t -eq "" -or $t.StartsWith("#") -or -not $t.Contains("=")) { continue }
    $name, $value = $t.Split("=", 2)
    $name = $name.Trim(); $value = $value.Trim().Trim('"').Trim("'")
    if ($Allowed -contains $name -and $value -ne "" -and -not [Environment]::GetEnvironmentVariable($name)) {
      [Environment]::SetEnvironmentVariable($name, $value)
      Write-Host "$name read from .env"
    }
  }
}

if (-not $env:AGENT_PROVIDER -and -not $env:A4AI_ANTHROPIC_API_KEY -and $env:GEMINI_API_KEY) {
  $env:AGENT_PROVIDER = "gemini"
  Write-Host "No Anthropic key: using Gemini (AGENT_PROVIDER=gemini)"
}

if ($Voice) {
  $env:VOICE_ENABLED = "1"
  if (-not $env:SARVAM_API_KEY) { Write-Warning "VOICE_ENABLED=1 but SARVAM_API_KEY is not set: the mic will stay hidden (not_configured)" }
  else { Write-Host "Voice on (VOICE_ENABLED=1)" }
}
