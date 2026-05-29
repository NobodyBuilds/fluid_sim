@echo off
cd /d "%~dp0"
if not exist build mkdir build
set CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"

echo.
echo ======================================
echo   CUDA Compilation Target
echo ======================================
echo   [1] RTX 2050 only  (sm_86, fast)
echo   [2] All targets    (sm_75 to sm_90)
echo ======================================
echo.

choice /c 12 /n /m "Select target [1/2]: "

if errorlevel 2 goto ALL_TARGETS
if errorlevel 1 goto SINGLE_TARGET

:SINGLE_TARGET
echo.
echo [TARGET] RTX 2050 only (compute_86 / sm_86)
echo.
set GENCODE=-gencode arch=compute_86,code=sm_86
goto COMPILE

:ALL_TARGETS
echo.
echo [TARGET] All architectures (sm_75 through sm_90)
echo.
set GENCODE=^
 -gencode arch=compute_75,code=sm_75 ^
 -gencode arch=compute_80,code=sm_80 ^
 -gencode arch=compute_86,code=sm_86 ^
 -gencode arch=compute_89,code=sm_89 ^
 -gencode arch=compute_90,code=sm_90 ^
 -gencode arch=compute_90,code=compute_90
goto COMPILE

:COMPILE
echo Compiling...
echo.
nvcc -c source\compute.cu -o source\compute.obj ^
 %GENCODE% ^
 --std=c++17 ^
 -O3 ^
 --use_fast_math ^
 -Xptxas=-O3,--allow-expensive-optimizations=true ^
 -Xptxas=-v ^
 -lineinfo ^
 -Xcompiler="/MD /O2 /fp:fast /Zc:preprocessor /permissive-" ^
 -I"D:\visual_studio\fluid_sim\glad\include" ^
 -I"D:\visual_studio\fluid_sim\glfw-3.4.bin.WIN64\include"

if %errorlevel% neq 0 (
    echo.
    echo [FAILED] nvcc exited with error %errorlevel%
    pause
    exit /b %errorlevel%
)

lib /OUT:build\compute.lib source\compute.obj
echo.
echo [OK] CUDA library built: build\compute.lib
pause