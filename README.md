# VisionTools

VisionRT 的工具实现与作者声明。职责和验证入口见 [文档索引](docs/README.md)。

- [CPU 模块](src/preprocess/PreprocessModule.cpp)、[推理模块](src/infer/InferModule.cpp)：显式注册稳定工具 ID。
- [JSON 作者声明](assets/tools)：端口、默认值、编辑器元数据及 Python 入口。
- [发行目录表](tools/package_catalog.json)：声明路径与模块归属；不复制声明中的身份或端口。
- [发行管线](tools/package_tools.py)：从声明和构建清单派生 module_index.json，调用宿主校验器后事务发布。
- [公共 SDK](3rdparty/VisionRTContracts)、[CMake presets](CMakePresets.json)。

## 构建

原生 CPU 构建只需要 CMake、C++20 编译器和 OpenCV。推理依赖见 [ConfigInferRT](cmake/ConfigInferRT.cmake)；Python 测试和打包使用标准库，不需要 PyYAML。

```cmd
git submodule update --init --recursive
tools\build.cmd cmake --preset cpu -DVISIONTOOLS_BUILD_PYTHON=OFF
tools\build.cmd cmake --build build --target visiontools_preprocess
tools\build.cmd ctest --test-dir build -R test_module_contract --output-on-failure
```

依赖路径使用 CMake 参数、环境变量或未跟踪的 CMakeUserPresets.json。打包所需宿主校验器通过 `VISIONTOOLS_VALIDATOR` 环境变量、同名 CMake 参数或打包器的 `--validator` 指定。

```cmd
python tools\package_tools.py --build-dir build --install-dir dist\VisionTools --validator <宿主校验器路径>
```

构建与安装步骤见 [CMake](CMakeLists.txt)，发行目录保护和回滚见 [定向测试](tests/test_package.py)。
