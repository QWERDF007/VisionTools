# VisionTools 工程工具集

本目录包含自动化构建、依赖管理、元数据生成与发布打包工具。

## 1. 核心文件说明

- `build.cmd`: 一站式开发与构建脚本，自动探测 MSVC x64 环境，支持增量构建、测试、清理和打包。
- `../CMakePresets.json`: `cpu` / `infer` 两个显式构建预设；本机依赖路径放在未跟踪的 `CMakeUserPresets.json`、`-D` 或环境变量中，tracked 配置不含个人路径。
- [作者声明与模块注册入口](../docs/README.md)、[声明与分发验证](../tests/test_distribution.py)：手写工具实现和注册，声明语义由真实宿主 CLI 校验。
- `package_tools.py`: 工具包安装收集与交付发布脚本，生成供 VisionRT 消费的相对资源清单；安装目录、ZIP 与外置索引作为同一发行组预检并事务提交。

## 2. 常用开发命令

```cmd
:: CPU 预设（不探测 InferRT 与 CUDA）
tools\build.cmd cmake --preset cpu
tools\build.cmd cmake --build build --config Release

:: 默认 Release 构建（含 InferRT，需要本机依赖路径）
tools\build.cmd

:: 编译并执行契约测试
tools\build.cmd --test

:: 清理并重新构建
tools\build.cmd --clean

:: 打包发布产物（安装目录、ZIP、外置索引）
tools\build.cmd --package dist\VisionTools

:: 在 MSVC x64 环境下透传任意命令
tools\build.cmd ctest --test-dir build --output-on-failure
```
