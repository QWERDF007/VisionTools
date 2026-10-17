@echo off
setlocal

set "PROJECT_ROOT=%~dp0.."
pushd "%PROJECT_ROOT%"

if defined VSCMD_VER goto :VSCMD_INITIALIZED

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" set "VSWHERE=%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe"

if not exist "%VSWHERE%" (
    echo [ERROR] Visual Studio Installer vswhere.exe not found.
    popd
    exit /b 1
)

for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do call "%%i\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 (
    echo [ERROR] Failed to initialize Visual Studio VC environment.
    popd
    exit /b 1
)

:VSCMD_INITIALIZED
setlocal enabledelayedexpansion

set "FIRST_ARG=%~1"

if "%FIRST_ARG%"=="help" goto :SHOW_HELP
if "%FIRST_ARG%"=="-h" goto :SHOW_HELP
if "%FIRST_ARG%"=="--help" goto :SHOW_HELP
if "%FIRST_ARG%"=="/?" goto :SHOW_HELP

if "%FIRST_ARG%"=="clean" goto :DO_BUILD
if "%FIRST_ARG%"=="test" goto :DO_BUILD
if "%FIRST_ARG%"=="" goto :DO_BUILD

set "CHAR1=%FIRST_ARG:~0,1%"
if "%CHAR1%"=="-" goto :DO_BUILD

popd
%*
exit /b %errorlevel%

:DO_BUILD
set "CONFIG=Release"
set "BUILD_DIR=build"
set "DO_CLEAN=0"
set "DO_TEST=0"
set "DO_PACKAGE=0"
set "BUILD_INFER=ON"
set "PACKAGE_DIR="

:PARSE_ARGS
if "%~1"=="" goto :START_BUILD
if /i "%~1"=="clean" (
    set "DO_CLEAN=1"
    shift
    goto :PARSE_ARGS
)
if /i "%~1"=="--clean" (
    set "DO_CLEAN=1"
    shift
    goto :PARSE_ARGS
)
if /i "%~1"=="test" (
    set "DO_TEST=1"
    shift
    goto :PARSE_ARGS
)
if /i "%~1"=="--test" (
    set "DO_TEST=1"
    shift
    goto :PARSE_ARGS
)
if /i "%~1"=="--config" (
    set "CONFIG=%~2"
    shift
    shift
    goto :PARSE_ARGS
)
if /i "%~1"=="--debug" (
    set "CONFIG=Debug"
    shift
    goto :PARSE_ARGS
)
if /i "%~1"=="--release" (
    set "CONFIG=Release"
    shift
    goto :PARSE_ARGS
)
if /i "%~1"=="--no-infer" (
    set "BUILD_INFER=OFF"
    shift
    goto :PARSE_ARGS
)
if /i "%~1"=="--infer" (
    set "BUILD_INFER=ON"
    shift
    goto :PARSE_ARGS
)
if /i "%~1"=="--build-dir" (
    set "BUILD_DIR=%~2"
    shift
    shift
    goto :PARSE_ARGS
)
if /i "%~1"=="--package" (
    set "DO_PACKAGE=1"
    if not "%~2"=="" (
        set "CHAR_TEST=%~2"
        set "CHAR_TEST=!CHAR_TEST:~0,1!"
        if not "!CHAR_TEST!"=="-" (
            set "PACKAGE_DIR=%~2"
            shift
        )
    )
    shift
    goto :PARSE_ARGS
)
echo [WARN] Unknown option: %~1, passing through as CMake argument.
shift
goto :PARSE_ARGS

:START_BUILD
if "%DO_CLEAN%"=="1" (
    echo [INFO] Cleaning build directory: %BUILD_DIR%
    if exist "%BUILD_DIR%" rd /s /q "%BUILD_DIR%"
)

echo [INFO] Configuring VisionTools (Config=%CONFIG%, Infer=%BUILD_INFER%, BuildDir=%BUILD_DIR%)...
cmake -B "%BUILD_DIR%" -G "Ninja" -DCMAKE_BUILD_TYPE=%CONFIG% -DVISIONTOOLS_BUILD_INFER=%BUILD_INFER% -DVISIONTOOLS_BUILD_TESTS=ON
if errorlevel 1 (
    echo [INFO] Ninja configuration failed, trying default generator...
    cmake -B "%BUILD_DIR%" -DCMAKE_BUILD_TYPE=%CONFIG% -DVISIONTOOLS_BUILD_INFER=%BUILD_INFER% -DVISIONTOOLS_BUILD_TESTS=ON
    if errorlevel 1 (
        echo [ERROR] CMake configuration failed.
        popd
        exit /b 1
    )
)

echo [INFO] Building VisionTools...
cmake --build "%BUILD_DIR%" --config %CONFIG%
if errorlevel 1 (
    echo [ERROR] Build failed.
    popd
    exit /b 1
)

if "%DO_TEST%"=="1" (
    echo [INFO] Running CTest...
    ctest --test-dir "%BUILD_DIR%" -C %CONFIG% --output-on-failure
    if errorlevel 1 (
        echo [ERROR] Some tests failed.
        popd
        exit /b 1
    )
)

if "%DO_PACKAGE%"=="1" (
    echo [INFO] Packaging tools...
    if "%PACKAGE_DIR%"=="" set "PACKAGE_DIR=dist\VisionTools"
    python tools\package_tools.py --config %CONFIG% --build-dir "%BUILD_DIR%" --install-dir "%PACKAGE_DIR%"
    if errorlevel 1 (
        echo [ERROR] Tool packaging failed.
        popd
        exit /b 1
    )
)

echo [SUCCESS] VisionTools build completed successfully.
popd
exit /b 0

:SHOW_HELP
echo Usage: tools\build.cmd [options]
echo.
echo Options:
echo   --release             Build in Release mode (default)
echo   --debug               Build in Debug mode
echo   --test, test          Build and run all unit/contract tests
echo   --clean, clean        Remove build directory before building
echo   --no-infer            Disable InferRT tools (CPU-only build)
echo   --infer               Enable InferRT tools (default: ON)
echo   --package [DIR]       Package artifacts for distribution/VisionRT consumption
echo   --build-dir DIR       Specify build directory (default: build)
echo   help, -h, --help      Show this help message
echo.
echo Or run any command inside MSVC x64 environment:
echo   tools\build.cmd cmake --build build
echo   tools\build.cmd ctest --test-dir build --output-on-failure
popd
exit /b 0
