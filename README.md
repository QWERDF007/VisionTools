# VisionTools

VisionRT 的工具库。普通 CMake 构建输出领域 DLL；显式打包输出一个 ZIP 与同源 `module_index.yaml`。发行配置名不约束计算后端或工具组合。

## 作者与实现入口

- [工具声明](assets/tools)：工具身份、分类、端口、Python 入口及显式资源的唯一作者源，包含平台接收节点声明。
- [实现绑定](tools/module_bindings.yaml)：C++ 工厂和宿主能力绑定。
- [预处理领域](src/preprocess)、[推理领域](src/infer)：多个工具共享领域 DLL，不要求每个工具一个 DLL。
- [生成器](tools/generate_tool_contracts.py)：CMake 按实际选中目标生成合同与中间索引。
- [打包器](tools/package_tools.py)：消费构建清单与声明；发布前使用 SDK 校验。
- [共享 SDK 子模块](3rdparty/VisionRTContracts)：与宿主复用相同数据模型及包读取规则。
- [CMake presets](CMakePresets.json)：`cpu` / `infer` 两个显式预设；本机依赖路径放在未跟踪的 `CMakeUserPresets.json`、`-D` 或环境变量中。

## Windows 构建与交付

依赖 CMake、MSVC、OpenCV 和带 PyYAML 的 Python；推理目标所需前提见对应 CMake 与依赖配置。依赖路径通过 `-DOpenCV_DIR=...`、环境变量或 `CMakeUserPresets.json` 提供，仓库内不保存个人机器路径。

```cmd
git submodule update --init --recursive
tools\build.cmd cmake --preset cpu
tools\build.cmd cmake --build build --config Release
tools\build.cmd --release --no-infer --package dist\VisionTools
tools\build.cmd ctest --test-dir build --output-on-failure
```

`--no-infer` 仅控制当前推理目标是否参与构建，不定义包类型。打包输出 `dist/VisionTools` 安装目录、`dist/VisionTools.zip` 与同源的 `dist/VisionTools.module_index.yaml`；安装目录带有本工具包标记，重复打包会先校验归属并保留旧包直到新包提交成功。宿主部署脚本以 `--tools-package` 消费 ZIP。

- [文档索引](docs/README.md)
- [唯一任务账本](WORKLOG.md)
