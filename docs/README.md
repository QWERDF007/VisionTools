# VisionTools 文档索引

- [公共值、状态与注册接口](../3rdparty/VisionRTContracts/docs/README.md)
- [CPU 注册](../src/preprocess/PreprocessModule.cpp)、[推理注册](../src/infer/InferModule.cpp)、[模块内构造与销毁](../src/ToolFactory.h)
- [作者声明](../assets/tools)、[发行目录表](../tools/package_catalog.json)
- [构建与安装](../CMakeLists.txt)、[依赖解析](../cmake/ConfigDependencyDefaults.cmake)、[预设](../CMakePresets.json)
- [作者 JSON 校验、发行事务与路径保护](../tools/package_tools.py)、[事务验证](../tests/test_package.py)
- [真实模块注册验证](../tests/test_module_contract.cpp)、[动态消费](../tests/test_installation_consumer.cpp)、[算法数值](../tests/test_preprocess_contract.cpp)
- [状态与原始异常详情](../3rdparty/VisionRTContracts/include/visionrt/contracts/Tool.h)、[图像解码边界](../src/preprocess/ReadImageTool.cpp)
- [声明与分发验证](../tests/test_distribution.py)、[Python 示例](../assets/tools/custom/Python)

## 新增工具

以 [AdaptiveBinaryTool](../src/preprocess/AdaptiveBinaryTool.h)、[实现](../src/preprocess/AdaptiveBinaryTool.cpp) 为最小原生示例；接入位置分别为 [模块注册](../src/preprocess/PreprocessModule.cpp)、[CMake 目标](../src/preprocess/CMakeLists.txt)、[JSON 声明](../assets/tools/builtin/预处理/图像处理/自适应二值化.json) 与 [发行目录表](../tools/package_catalog.json)。工具只消费公共 Values，不解析作者声明。

Python 示例的入口目录即代码包根；声明的 resources 才会收集非代码资源。实际规则和失败反例见 [打包器](../tools/package_tools.py) 与 [测试](../tests/test_package.py)，不另存第二份规则。
