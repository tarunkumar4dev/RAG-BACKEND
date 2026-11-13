@echo off
REM a4ai Agent on REAL a4ai data, READ-ONLY (Windows Batch).
REM Run after agent/sql/create_agent_ro_role.sql
REM Usage: scripts\run_agent_real.bat
REM Will prompt for agent_ro password, Anthropic key, Gemini key, and project details

setlocal enabledelayedexpansion

set "ROOT=%~dp0.."
cd /d "%ROOT%"

set "Py=%ROOT%\agent\.devdata\venv\Scripts\python.exe"
if not exist "%Py%" (
    echo Error: Python venv not found at %Py%
    echo Run scripts\run_agent_dev.ps1 once first to create the venv
    exit /b 1
)

REM Remove dev mode variables if set
set "AGENT_DEV_MODE="
set "AGENT_DEV_AUTH="
set "AGENT_DB="

REM Prompt for ProjectRef
set /p ProjectRef="Enter Supabase Project Ref: "
if "!ProjectRef!"=="" (
    echo Error: Project Ref is required
    exit /b 1
)

REM Prompt for Region (default ap-south-1)
set /p Region="Enter AWS Region (default: ap-south-1): "
if "!Region!"=="" set "Region=ap-south-1"

REM Prompt for Port (default 8000)
set /p Port="Enter Port (default: 8000): "
if "!Port!"=="" set "Port=8000"

REM Prompt for agent_ro password (plain text since batch doesn't have built-in masking)
set /p AgentRoPassword="Enter agent_ro password: "
if "!AgentRoPassword!"=="" (
    echo Error: agent_ro password is required
    exit /b 1
)

REM URL encode the password (basic replacement of special chars)
set "EncPassword=!AgentRoPassword!"
REM Note: Full URL encoding would require VBScript or external tool; this is a simplified version
call set "EncPassword=!EncPassword:@=%%40!"
call set "EncPassword=!EncPassword: =%%20!"

echo Setting up environment variables...

echo.  AGENT_REAL_DB_URL
set "AGENT_REAL_DB_URL=postgresql://agent_ro.!ProjectRef!:!EncPassword!@aws-0-!Region!.pooler.supabase.com:5432/postgres?sslmode=require"

echo.  AGENT_REAL_DB_MODE
set "AGENT_REAL_DB_MODE=1"

echo.  SUPABASE_URL
set "SUPABASE_URL=https://!ProjectRef!.supabase.co"

echo.  AGENT_CORS_ORIGINS
set "AGENT_CORS_ORIGINS=http://localhost:5173,http://localhost:8080"

REM Check if Anthropic key is set in env
if not defined A4AI_ANTHROPIC_API_KEY (
    set /p A4AI_ANTHROPIC_API_KEY="Enter A4AI_ANTHROPIC_API_KEY: "
) else (
    echo.  A4AI_ANTHROPIC_API_KEY (from environment)
)

REM Check if Gemini key is set in env
if not defined GEMINI_API_KEY (
    set /p GEMINI_API_KEY="Enter GEMINI_API_KEY (optional, leave blank to skip): "
)
if defined GEMINI_API_KEY (
    echo.  GEMINI_API_KEY
) else (
    echo.  GEMINI_API_KEY (not set)
)

REM Check if JWT secret is set
if not defined SUPABASE_JWT_SECRET (
    echo SUPABASE_JWT_SECRET not set: tokens will be verified with the project's JWKS
    echo If your project uses legacy JWT secret, set it before running this script
) else (
    echo.  SUPABASE_JWT_SECRET (from environment)
)

REM Kill any existing process on the port
echo Checking for existing process on port !Port!...
for /f "tokens=5" %%a in ('netstat -aon ^| find "LISTENING" ^| find ":!Port!"') do (
    echo Killing process on port !Port! (PID: %%a)
    taskkill /PID %%a /F >nul 2>&1
)

echo.
echo Starting a4ai Agent (LIVE, read-only) on http://localhost:!Port!
"%Py%" -m uvicorn agent.api:app --host 127.0.0.1 --port !Port!

set ExitCode=!ERRORLEVEL!

REM Clear sensitive env var
set "AGENT_REAL_DB_URL="

exit /b !ExitCode!
