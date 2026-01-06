@echo off
setlocal

set "ROOT=%~dp0.."
set "PS1=%~dp0build.ps1"

if not exist "%PS1%" (
  echo build.ps1 not found at "%PS1%"
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%PS1%"
