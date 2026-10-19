# VisionTools

VisionRT 的工具库。普通 CMake 构建输出领域 DLL；显式打包输出一个 ZIP 与同源 `module_index.yaml`。发行配置名不约束计算后端或工具组合。

## 作者与实现入口

- [工具声明](assets/tools)：工具身份、分类、端口、Python 入口及显式资源的唯一作者源，包含平台接收节点声明。
- [实现绑定](tools/module_bindings.yaml)：C++ 工厂和宿主能力绑定。
- [预处理领域](src/preprocess)、[推理领域](src/infer)：多个工具共享领域 DLL，不要求每个工具一个 DLL。
- [生成器](tools/generate_tool_contracts.py)：CMake 按实际选中目标生成合同与中间索引。
- [打包器](tools/package_tools.py)：消费构建清单与声明；发布前使用 SDK 校验。
- [共享 SDK 子模块](3rdparty/VisionRTSDK)：与宿主复用相同数据模型及包读取规则。
- [依赖配置](tools/dependencies.yaml)、[CMake](CMakeLists.txt)：本机路径与实际构建选择。

## Windows 构建与交付

依赖 CMake、MSVC、OpenCV 和带 PyYAML 的 Python；推理目标所需前提见对应 CMake 与依赖配置。

```cmd
git submodule update --init --recursive
tools\build.cmd --release --no-infer
tools\build.cmd --release --no-infer --package dist\VisionTools
tools\build.cmd ctest --test-dir build --output-on-failure
```

`--no-infer` 仅控制当前推理目标是否参与构建，不定义包类型。需要该目标时使用已配置的默认构建；包内容由实际产物决定。打包输出 `dist/VisionTools.zip` 和 `dist/module_index.yaml`，宿主部署脚本以 `--tools-package` 消费 ZIP。

- [文档索引](docs/README.md)
- [唯一任务账本](WORKLOG.md)
