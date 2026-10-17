# VisionTools 工程工具集

本目录包含自动化构建、依赖管理、元数据生成与发布打包工具。

## 1. 核心文件说明

- `build.cmd`: 一站式开发与构建脚本，自动探测 MSVC x64 环境，支持增量构建、测试、清理和打包；
- `dependencies.yaml`: 依赖单一真相源（SSOT），集中记录 OpenCV, InferRT, CUDA, TensorRT, Python 等默认基准路径；
- `generate_tool_contracts.py`: 契约生成器，解析算子 YAML 生成 C++ 端口常量、TypeSpec 校验与领域模块工厂映射；
- `package_tools.py`: 工具包安装收集与交付发布脚本，生成供 VisionRT 消费的相对资源清单；
- `link_dependencies.py`: 辅助脚本，开发阶段快速软链接运行时依赖 DLL 至构建输出目录。

## 2. 常用开发命令

```cmd
:: 默认 Release 构建（含 InferRT）
tools\build.cmd

:: 仅 CPU 模式构建（不探测 InferRT 与 CUDA）
tools\build.cmd --no-infer

:: 编译并执行契约测试
tools\build.cmd --test

:: 清理并重新构建
tools\build.cmd --clean

:: 打包发布产物
tools\build.cmd --package dist\VisionTools

:: 在 MSVC x64 环境下透传任意命令
tools\build.cmd ctest --test-dir build --output-on-failure
```
