@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=
set ATTN_PRECISION=fp16

@REM call webui.bat --help
call webui.bat --xformers --ckpt-dir "D:\Stable-Diffusion"