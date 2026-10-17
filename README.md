# VisionTools

VisionTools 是 VisionRT 视觉平台的算法与工具库，按领域划分为独立的动态链接库与声明规范。

## 领域模块与结构

- **声明单一真相源 (SSOT)**: [assets/tools](assets/tools)
  - 内置工具声明：[assets/tools/builtin](assets/tools/builtin)
  - 示例自定义脚本：[assets/tools/custom](assets/tools/custom)
- **实现绑定映射**: [tools/module_bindings.yaml](tools/module_bindings.yaml)
- **契约生成工具**: [tools/generate_tool_contracts.py](tools/generate_tool_contracts.py)（由 CMake 自动驱动生成端口常量、TypeSpec、工厂派发及模块索引）
- **预处理模块 (CPU)**: [src/preprocess](src/preprocess) (`visiontools_preprocess.dll`)
  - 图像读取：`visionrt.image.read`
  - ROI 投影：`visionrt.roi.projection`
  - 矩形无拷贝只读裁剪：`visionrt.image.crop`
  - 图像滤波预处理：`visionrt.image.preprocess`
  - 自适应二值化：`visiontools.preprocess.adaptive_binary`
  - 缺陷与目标过滤：`visionrt.detection.filter`
- **推理模块 (GPU)**: [src/infer](src/infer) (`visiontools_infer.dll`)
  - YOLO 目标检测：`visionrt.dl.yolo`
- **推理探针与基线验证**: [probes/infer](probes/infer) (`visiontools_infer_probe.exe`)
- **外部依赖配置**: [tools/dependencies.yaml](tools/dependencies.yaml)

## 外部依赖

- **VisionRTToolSDK**: 外部安装的 SDK 单一真相源，提供 `VisionRT::ToolSDK` 目标与共享 ABI 合同（禁止仓库内自建副本）。
- **OpenCV**: OpenCV 4.8.0+。
- **InferRT / CUDA / TensorRT**: 仅 GPU 推理模块消费，CPU 模式无需安装。
- **Python**: 3.9+，构建时依赖 `PyYAML`。

## 构建与测试指南

Windows MSVC x64 环境下可通过统一构建脚本 [tools/build.cmd](tools/build.cmd) 完成完整生命周期：

```cmd
# 1. 完整 GPU 构建并运行所有 CTest 契约与单元测试
tools\build.cmd --clean --test

# 2. 纯 CPU 环境构建并运行测试 (无 InferRT / CUDA 依赖)
tools\build.cmd -B build-cpu --no-infer --clean --test

# 3. 打包 GPU 发布包并自动验证闭包完整性
tools\build.cmd --package dist\VisionTools_GPU

# 4. 打包 CPU 发布包并自动验证无 GPU 污染
tools\build.cmd -B build-cpu --no-infer --package dist\VisionTools_CPU
```

## 文档与任务追踪

- 文档索引与复用记录：[docs/README.md](docs/README.md)
- 唯一任务账本：[WORKLOG.md](WORKLOG.md)
