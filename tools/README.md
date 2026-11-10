# VisionTools 工程工具集

本目录包含自动化构建、依赖管理、元数据生成与发布打包工具。

## 1. 核心文件说明

- `build.cmd`: 一站式开发与构建脚本，自动探测 MSVC x64 环境，支持增量构建、测试、清理和打包。
- `../CMakePresets.json`: `cpu` / `infer` 两个显式构建预设；本机依赖路径放在未跟踪的 `CMakeUserPresets.json`、`-D` 或环境变量中，tracked 配置不含个人路径。
- [作者声明与模块注册入口](../docs/README.md)、[声明与分发验证](../tests/test_distribution.py)：手写工具实现和注册，声明语义由真实宿主 CLI 校验。
- `package_tools.py`: 工具包安装收集与交付发布脚本，生成供 VisionRT 消费的相对资源清单；安装目录、ZIP 与外置索引作为同一发行组预检并事务提交。
- `release_check.py`: 发行前的宿主消费检查。设置 `VISIONTOOLS_REQUIRE_HOST_EXECUTION=1` 并要求 VisionRT 的消费者、部署脚本与校验器存在；缺任一项直接失败，不会退回 skip。

## 2. 常用开发命令

```cmd
:: CPU 预设（不探测 InferRT 与 CUDA）
tools\build.cmd cmake --preset cpu
tools\build.cmd cmake --build build --config Release

:: 默认 Release 构建（含 InferRT，需要本机依赖路径）
tools\build.cmd

:: 编译并执行契约测试（测试可执行不进默认构建，--test 会先构建 visiontools_tests）
tools\build.cmd --test

:: 只构建全部测试可执行，不运行
tools\build.cmd cmake --build build --target visiontools_tests

:: 清理并重新构建
tools\build.cmd --clean

:: 打包发布产物（安装目录、ZIP、外置索引）
tools\build.cmd --package dist\VisionTools

:: 在 MSVC x64 环境下透传任意命令（直接跑 ctest 前需已构建 visiontools_tests）
tools\build.cmd cmake --build build --target visiontools_tests
tools\build.cmd ctest --test-dir build --output-on-failure

:: 发行检查：把真实宿主借给分发测试，缺宿主即失败（不是 skip）
python tools\release_check.py
python tools\release_check.py tests.test_distribution.DistributionContract.test_python_package_executes_in_real_host
```

`release_check.py` 默认取同级 `../VisionRT/build/bin/python_tools_execution_test.exe`、`../VisionRT/tools/package_app.py` 与 `../VisionRT/build/bin/visionrt_validate_tool.exe`，可用 `--consumer`、`--deploy-script`、`--validator` 覆盖。它把三个路径同时写入环境变量并强制 `VISIONTOOLS_REQUIRE_HOST_EXECUTION=1`，因此“打包成功但宿主跑不起来”会在这里暴露，而不是留到用户机器上。
