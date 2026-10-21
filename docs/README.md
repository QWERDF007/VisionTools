# VisionTools 文档索引与架构设计

本文档作为 VisionTools 仓库的唯一文档导航入口，遵循单一真相源（SSOT）原则，指向实际代码与配置文件，避免文字复述形成分叉。

## 导航索引

- **项目概览与构建使用**: [../README.md](../README.md)
- **唯一任务账本与验证记录**: [../WORKLOG.md](../WORKLOG.md)
- **工具声明目录 (SSOT)**: [../assets/tools/](../assets/tools/)
- **矩形裁剪**: [作者声明](../assets/tools/builtin/预处理/图像处理/图像裁剪.yaml)、[算子实现](../src/preprocess/ImageCropTool.cpp)、[执行契约](../tests/test_preprocess_contract.cpp)、[真实 DLL 消费](../tests/test_installation_consumer.cpp)。
  - 内置算子：`assets/tools/builtin/`
  - Python 工具：`assets/tools/custom/`；必要非代码资源在对应声明的 resources 中显式列出。
- **作者命名验收**: [test_distribution.py](../tests/test_distribution.py)；领域 DLL 实际输出名见[图像目标](../src/preprocess/CMakeLists.txt)与[推理目标](../src/infer/CMakeLists.txt)。
- **单一跨流程接收声明**: [跨流程接收.yaml](../assets/tools/builtin/流程控制/跨流程接收.yaml)；类型引用的公共定义与校验见 SDK [Schema.h](../3rdparty/VisionRTContracts/include/visionrt/contracts/Schema.h)。
- **模块绑定配置**: [../tools/module_bindings.yaml](../tools/module_bindings.yaml)
- **依赖路径默认值**: [../tools/dependencies.yaml](../tools/dependencies.yaml)
- **原生运行库收集**: [CMake 安装组件](../CMakeLists.txt)；打包入口消费该组件的实际安装清单，动态加载的后端资源由同一组件安装。
- **代码生成与工厂接入**: [../tools/generate_tool_contracts.py](../tools/generate_tool_contracts.py)
- **分发打包与闭包校验**: [../tools/package_tools.py](../tools/package_tools.py)
- **构建驱动脚本**: [../tools/build.cmd](../tools/build.cmd)
- **发行链测试**: [../tests/test_distribution.py](../tests/test_distribution.py)；设置 VISIONRT_PYTHON_CONSUMER 为宿主构建的 python_contract.exe、VISIONRT_DEPLOY_SCRIPT 为宿主 package_app.py 时，启用真实跨仓 ZIP 部署与执行验证。

## 复用决策记录

| 需求领域 | 复用实现 / 库 | 应用方式与理由 |
|---|---|---|
| 图像预处理与阈值计算 | [OpenCV](https://github.com/opencv/opencv) (`imgproc`, `core`, `imgcodecs`) | 直接调用 `cv::adaptiveThreshold`、`cv::imread` 等成熟接口；矩形区域裁剪采用 `image.region()` 实现零拷贝视图并共享 OpenCV 内存块 `u`，保留 `rowStride`。 |
| 声明解析与校验 | [PyYAML](https://github.com/yaml/pyyaml) | 严格依赖标准 PyYAML safe_load，消除私有简化解析带来的嵌套结构解析错误与类型歧义。 |
| 深度学习推理与显存管理 | [InferRT](https://github.com/QWERDF007/InferRT) / TensorRT | PRIVATE 方式链接 `InferRT::model`，公开合同与 ModuleApi 不泄露后端类型；在 GPU 传输完成后同步并将结果封装为独立的 CPU 结构。 |
| 共享工具合同 | [VisionRTContracts 子模块](../3rdparty/VisionRTContracts) (`VisionRT::Contracts`) | 复用 SDK 的值、模块与包校验，不在生产者维护另一套端口解析语义。 |

## 契约与接口设计

- **统一入口**: 模块统一导出 `extern "C" const visionrt::ModuleApi* VT_CALL visiontoolsModuleApi() noexcept`。
- **工厂派发**: 由 `generate_tool_contracts.py` 根据 `module_bindings.yaml` 自动生成对应的 `visiontools_*_factory.inc`，杜绝手写注册漏项。
- **端口常量**: 由 YAML 声明自动提取生成 `PortConstants.h`，各算子实现严格引用命名空间常量。
