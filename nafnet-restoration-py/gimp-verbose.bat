@echo off
REM gimp-verbose.bat - launch GIMP with --verbose so plug-in stdout/stderr
REM is visible in the terminal that started GIMP. Use for debugging
REM the plug-in: progress markers, error messages, and stack traces
REM are printed to the console that launched this script.
REM
REM Usage:
REM   gimp-verbose.bat
REM   gimp-verbose.bat C:\Path\To\Blah.xcf

if defined GIMP_EXE_PATH (
    "%GIMP_EXE_PATH%" --verbose %*
) else (
    "%LOCALAPPDATA%\Programs\GIMP 3\bin\gimp-3.2.exe" --verbose %*
)
