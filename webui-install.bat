@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=

@REM call webui.bat install --reinstall-xformers --xformers --reinstall-torch
call webui.bat install --xformers
