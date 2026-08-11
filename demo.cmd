@echo off
setlocal
rem One-command demo entry point for Windows.
rem
rem This wrapper exists because PowerShell refuses to run demo.ps1 under the
rem Restricted execution policy (the Windows client default), and under
rem RemoteSigned if the repo was downloaded as a ZIP rather than cloned - a ZIP
rem carries the mark-of-the-web, which RemoteSigned rejects for unsigned
rem scripts. Passing -ExecutionPolicy Bypass applies to this invocation only;
rem it changes nothing on the machine.
rem
rem   demo            start everything
rem   demo -Setup     fetch corpus + build indices, then start
rem   demo -Check     preflight only
rem   demo -NoUi      API only

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0demo.ps1" %*
set "RC=%ERRORLEVEL%"

rem Hold the window open only when something went wrong, so a double-click that
rem fails preflight stays readable. A normal run ends via Ctrl+C and exits
rem straight away. With stdin redirected, pause returns immediately, so this
rem does not wedge scripted use.
if not "%RC%"=="0" pause

exit /b %RC%
