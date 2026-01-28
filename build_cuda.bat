@echo off

REM --- Force working directory to this script's folder ---
cd /d "%~dp0"

REM --- Ensure build directory exists ---
if not exist build mkdir build

REM --- CUDA path (optional, kept as you had it) ---
set CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"

REM --- Compile CUDA to object in build folder ---
nvcc -c source\compute.cu -o source\compute.obj ^ -O3 ^ --use_fast_math ^ --restrict ^ -Xptxas -O3 ^ -Xptxas --maxrregcount=64 ^ -Xcompiler "/MD /O2" ^ -arch=sm_86 ^ --extra-device-vectorization
REM --- Create static library from object ---
lib /OUT:build\compute.lib source\compute.obj

echo CUDA library built: build\compute.lib
pause
