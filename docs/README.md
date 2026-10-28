# VisionTools 文档索引

- [公共值、状态与注册接口](../3rdparty/VisionRTContracts/docs/README.md)
- [CPU 注册](../src/preprocess/PreprocessModule.cpp)、[推理注册](../src/infer/InferModule.cpp)、[模块内构造与销毁](../src/ToolFactory.h)
- [作者声明](../assets/tools)、[发行目录表](../tools/package_catalog.json)
- [构建与安装](../CMakeLists.txt)、[依赖解析](../cmake/ConfigDependencyDefaults.cmake)、[预设](../CMakePresets.json)
- [作者 JSON 与资源收集](../tools/package_payload.py)、[双来源与路径边界验证](../tests/test_package_payload.py)、[发行事务与目标保护](../tools/package_tools.py)、[事务验证](../tests/test_package.py)
- [真实模块注册验证](../tests/test_module_contract.cpp)、[动态消费](../tests/test_installation_consumer.cpp)、[算法数值](../tests/test_preprocess_contract.cpp)
- [分工具数值与边界用例](../tests/preprocess)、[裁剪独立像素金标与资源存活](../tests/preprocess/image_crop.cpp)、[真实 Mat 分配观测](../tests/preprocess/MatAllocationProbe.h)、[读图预算边界](../tests/preprocess/read_image.cpp)、[共享裁剪视图及外部保活](../tests/test_writable_tools_contract.cpp)
- [状态与原始异常详情](../3rdparty/VisionRTContracts/include/visionrt/contracts/Tool.h)、[图像解码边界](../src/preprocess/ReadImageTool.cpp)
- [C++ 参数子类](../3rdparty/VisionRTContracts/include/visionrt/contracts/Parameters.h)、[读图参数示例](../src/preprocess/ReadImageParameters.h)、[函数工具示例](../src/preprocess/BrightnessTool.h)、[Python 作者 SDK（唯一来源）](../python/author_sdk/visiontools_sdk/__init__.py)、[SDK 消费示例](../python/custom/examples/image_add/tool.py)
- [声明与分发验证](../tests/test_distribution.py)、[Python 内建源码](../python/builtin)、[Python 自定义源码](../python/custom)、[Python 直接调用与真实 NumPy 契约](../tests/test_python_tools.py)

## 新增工具

以 [AdaptiveBinaryTool](../src/preprocess/AdaptiveBinaryTool.h)、[实现](../src/preprocess/AdaptiveBinaryTool.cpp) 为最小原生示例；接入位置分别为 [模块注册](../src/preprocess/PreprocessModule.cpp)、[CMake 目标](../src/preprocess/CMakeLists.txt)、[JSON 声明](../assets/tools/builtin/预处理/图像处理/自适应二值化.json) 与 [发行目录表](../tools/package_catalog.json)。工具只消费公共 Values，不解析作者声明。

Python 来源、入口及资源规则见 [资源收集器](../tools/package_payload.py) 与 [测试](../tests/test_package.py)。

Python 工具以 `from .visiontools_sdk import ...` 使用作者 SDK。仓库里只有 [一个 SDK 源文件](../python/author_sdk/visiontools_sdk/__init__.py)；出包时由 [资源收集器](../tools/package_payload.py) 把它写进每个 Python 工具包，工具目录本身不再保存副本，也不修改 `sys.path`。

## 目标 profile 与出包集合

profile 决定模块、依赖库与工具三者的取舍，唯一的策略实现在 [package_profile](../tools/package_profile.py)，配置在 [package_profiles.json](../tools/package_profiles.json)：`cpu`（无推理）、`rtx40-runtime`（只加载引擎）、`rtx40-build`（保留现场构建）、`gpu-custom`（显式 `--sm` 集合）。

- 被 profile 排除的模块，其库文件与被依赖工具同批剔除；运行时清单不会再把它带回来。
- `engine_build` 的 profile 必须为目标架构提供 TensorRT builder 资源，缺失即失败，不发布无法构建引擎的包。
- 安装目录、ZIP 与索引在一次事务内提交；安装阶段的临时目录从创建起就在 `finally` 覆盖范围内，任何失败都保留旧产物。
