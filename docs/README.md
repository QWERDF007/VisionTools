# VisionTools 文档索引与架构设计

本文档作为 VisionTools 仓库的唯一文档导航入口，遵循单一真相源（SSOT）原则，指向实际代码与配置文件，避免文字复述形成分叉。

## 导航索引

- **项目概览与构建使用**: [../README.md](../README.md)
- **唯一任务账本与验证记录**: [../WORKLOG.md](../WORKLOG.md)
- **工具声明目录 (SSOT)**: [../assets/tools/](../assets/tools/)
  - 内置算子：`assets/tools/builtin/`
  - 示例脚本：`assets/tools/custom/`
- **模块绑定配置**: [../tools/module_bindings.yaml](../tools/module_bindings.yaml)
- **运行依赖配置**: [../tools/dependencies.yaml](../tools/dependencies.yaml)
- **代码生成与工厂接入**: [../tools/generate_tool_contracts.py](../tools/generate_tool_contracts.py)
- **分发打包与闭包校验**: [../tools/package_tools.py](../tools/package_tools.py)
- **构建驱动脚本**: [../tools/build.cmd](../tools/build.cmd)

## 复用决策记录

| 需求领域 | 复用实现 / 库 | 应用方式与理由 |
|---|---|---|
| 图像预处理与阈值计算 | [OpenCV](https://github.com/opencv/opencv) (`imgproc`, `core`, `imgcodecs`) | 直接调用 `cv::adaptiveThreshold`、`cv::imread` 等成熟接口；矩形区域裁剪采用 `image.region()` 实现零拷贝视图并共享 OpenCV 内存块 `u`，保留 `rowStride`。 |
| 声明解析与校验 | [PyYAML](https://github.com/yaml/pyyaml) | 严格依赖标准 PyYAML safe_load，消除私有简化解析带来的嵌套结构解析错误与类型歧义。 |
| 深度学习推理与显存管理 | [InferRT](https://github.com/QWERDF007/InferRT) / TensorRT | PRIVATE 方式链接 `InferRT::model`，公开合同与 ModuleApi 不泄露后端类型；在 GPU 传输完成后同步并将结果封装为独立的 CPU 结构。 |
| 外部宿主 SDK 抽象 | VisionRTToolSDK (`VisionRT::ToolSDK`) | 严格以 REQUIRED 方式消费外部 SDK（VisionRT 构建树导出包或安装包），使用统一的 `visionrt::Value`、`visionrt::ModuleApi`，不保留任何在树 SDK 分叉副本。 |

## 契约与接口设计

- **统一入口**: 模块统一导出 `extern "C" const visionrt::ModuleApi* VT_CALL visiontoolsModuleApi() noexcept`。
- **工厂派发**: 由 `generate_tool_contracts.py` 根据 `module_bindings.yaml` 自动生成对应的 `visiontools_*_factory.inc`，杜绝手写注册漏项。
- **端口常量**: 由 YAML 声明自动提取生成 `PortConstants.h`，各算子实现严格引用命名空间常量。
