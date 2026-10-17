# WORKLOG

> 本文件是项目唯一的任务账本。真实日志按最新在前追加在固定示例条目之后，并固定位于其他真实日志之上；`⏳ 待你裁决` 始终固定在顶部。

## ⏳ 待你裁决

（暂无待裁决事项）

---

## 日志

<!--
建议格式：

### YYYY-MM-DD — 简短任务名

**目标**
- ...

**当前状态**
- 已完成：...
- 未完成：...

**验证证据**
- `command ...` → 关键结果
- 未验证项请明确写“未验证”

**下一步**
- ...
-->

### 2026-10-17 — VisionTools 代码复验整改与下一阶段交付 (VT2-01 ~ VT2-08)

**目标**
- 依照《VisionTools 代码复验与下一阶段方案 (01c3795)》（N01 ~ N10，VT2-01 ~ VT2-08，NS-01 ~ NS-08，R01 ~ R10）完成第二轮复验问题修复与闭环验证。
- 根除已提交的 `src/generated` 副本与 `__has_include` 回退逻辑，编译期严格单源消费构建目录产物 (VT2-01 / NS-01)。
- 深度集成 SDK schema 枚举，生成完整 `TypeSpec` 与契约结构；`YoloModel.cpp` 全面接线生成常量；强化生成器负面静态检查 (VT2-02 / NS-02)。
- 修复 `ImageCropTool` 采样中心覆盖判决 (R09)；`DetectionFilterTool` 严格限定有限 double 浮点；补齐 `ReadImageTool` 契约测试 (VT2-03 / NS-03)。
- 修复 `package_tools.py` 幽灵导入与目录删除安全 (R10)；闭包校验增加针对 module_dll、GPU 隔离与 OpenCV/InferRT/CUDA 的独立物理文件校验 (R06~R08)；依赖解析失败严格报错 (VT2-04 / NS-04)。
- 增加动态加载消费者测试 `test_installation_consumer`；强化单实例 100 次循环与 20 次生命周期循环测试及延迟与 RSS 统计 (VT2-05 / NS-05)。
- 探针解耦独立配置能力 (VT2-06 / NS-06)。
- 联合宿主验证与日志/规范闭环 (VT2-07, VT2-08)。

**当前状态**
- 已完成：
  - VT2-01 (NS-01 / N01): 删除 `src/generated` 纳入 `.gitignore`；移除所有模块内 `__has_include` 兜底；头文件包含路径严格指向 `${CMAKE_BINARY_DIR}/generated`。
  - VT2-02 (NS-02 / N02 / N03): `generate_tool_contracts.py` 深度对齐 SDK 导出枚举并生成 `TypeSpec` 规范契约结构与 `getToolContract()`；`YoloModel.cpp` 端口接线完全统一至 `visiontools::ports::visionrt_dl_yolo`；增加未定义绑定报错、工具目录为空报错、cross-module 重复注册拦截、bool rank 拦截、单边 range 默认值违例拦截、Python 入口文件与符号静态 AST 校验；`test_generator.py` 扩充至 17 项测试。
  - VT2-03 (NS-03 / N04 / N08): `ImageCropTool` 引入 `coversEveryPixelCenter` 严格判定 ROI 覆盖所有像素中心方走 `image.region(bounds)` 零拷贝视图，小数/非对齐坐标走掩码路径；`DetectionFilterTool` 强制严格 double（拒绝 Int、Bool、String、NaN/Inf）；`test_preprocess_contract` 补齐小数裁剪与 `ReadImageTool` 全量契约测试。
  - VT2-04 (NS-04 / N05 / N06 / N07): `package_tools.py` 移除无效导入与 modules-only 回退；引入 `.tmp_pkg_VisionTools_<pid>` 临时暂存目录，打包校验成功后再原子替换，防御打包到工程根目录并保护既有安装目录；`verify_package` 逐项核验 `tools[*].module_dll`、CPU 包 GPU DLL 隔离、以及 OpenCV/InferRT/CUDA 物理文件独立存在；修复 `tools/build.cmd` 中 `!PACKAGE_DIR!` 延迟变量展开；`ConfigDependencyDefaults.cmake` 与 `ConfigVisionRTToolSDK.cmake` 遇到显式无效路径严格报错退出。
  - VT2-05 (NS-05 / N09): 新增 `tests/test_installation_consumer.cpp`，纯动态加载 (`LoadLibraryA` + `GetProcAddress`)，不链接算子实现库；`test_module_contract.cpp` 增加单实例 100 次循环、20 次 create/destroy 循环、double release 幂等与 RSS/延迟指标输出。
  - VT2-06 (NS-06 / N10): `probes/infer/CMakeLists.txt` 支持独立外部 CMake 配置构建。
  - VT2-08 (NS-08): 规范文档、真实记录验证证据，分阶段提交至 `origin/dev`。
- 未完成（客观外部资源受限）：
  - VT2-06: YOLO 真实权重实测未验证（当前环境无 `.wts` 权重文件，代码接线与探针已通过构建及模块调用测试）。
  - VT2-07: 宿主联合加载未验证（当前独立仓库无外部宿主运行时环境；动态消费者测试与契约测试已独立通过）。

**验证证据**
- `python docs\VisionTools_代码复验与下一阶段方案_01c3795\VisionTools_复验交付\evidence\reproduce.py .`
  - 10 项缺陷复现用例 (R01~R10) 全部验证通过 (exit code 0)：
    - R01: missing bindings 严格拦截退出 (code 1)
    - R02: TypeSpec present= True
    - R03: 编译头文件仅包含 build/generated/PortConstants.h
    - R04: 单边 range 默认值违例、bool rank、缺失 Python 文件均被 REJECTED
    - R05: module_index.json 具备规范 tools 字典并验证通过
    - R06: 假 DLL 索引被 verify_package 拦截 (verify=False)
    - R07: CPU 索引包含 GPU DLL 被拦截 (verify=False)
    - R08: 缺失 OpenCV 被独立物理文件核验拦截 (verify=False)
    - R09: 浮点 ROI (0.75, 1.0) 与 (0.1, 0.1) 正确切换 mask / view
    - R10: 打包失败预留既有安装目录完整 (previous release preserved= True)
- `tools\build.cmd --clean --test`
  - GPU 模式编译成功，CTest 7/7 全部通过 (100% tests passed):
    - test_preprocess_contract: Passed (0.06 sec)
    - test_module_contract: Passed (0.08 sec)
    - test_installation_consumer: Passed (0.06 sec)
    - test_infer_contract: Passed (0.05 sec)
    - test_generator: Passed (0.31 sec, 17/17 tests)
    - test_package: Passed (0.55 sec, 9/9 tests)
    - test_python_tools: Passed (0.19 sec, 4/4 tests)
- `tools\build.cmd -B build-cpu --no-infer --clean --test`
  - CPU 独立模式编译成功，CTest 6/6 全部通过。
- `tools\build.cmd --package`
  - 成功打包 52 个文件至 `dist/VisionTools`，`verify_package` 通过 (variant=gpu)。
- `tools\build.cmd -B build-cpu --no-infer --package`
  - 成功打包 19 个文件至 `dist/VisionTools`，`verify_package` 通过 (variant=cpu)。
- `.\build-cpu\tests\test_installation_consumer.exe .\dist\VisionTools\bin\visiontools_preprocess.dll`
  - 动态消费者直接加载测试通过并输出 `[PASS] test_installation_consumer passed successfully.`。
- `.\build-cpu\tests\test_module_contract.exe`
  - 单实例 100 次循环与 20 次销毁重建通过，输出 `[METRICS] 100-run: p50=6.6us, p95=7.2us`。

**下一步**
- 推送变更至 `origin/dev`，交付成果等待复审。

### 2026-10-17 — VisionTools 验收与改进方案全面整改落地 (VT-01 ~ VT-08)

**目标**
- 针对《VisionTools 验收结果》与下一步规格方案包（F01 ~ F09，VT-01 ~ VT-08，VS-01 ~ VS-08）完成全量缺陷修复、架构规范化与闭环验证。
- 根除 SDK 本地代码分叉与 fallback，建立单源外部消费 (VT-01)。
- 建立 YAML + 绑定驱动的代码生成链单一注册入口，CMake 自动联动 (VT-02)。
- 优化 CPU 算子契约：轴对齐矩形实现零拷贝只读视图（共享底层内存），严格类型约束 (VT-03)。
- 修复推理探针 CMake 与代码，消费生产模块与已安装 SDK (VT-04)。
- 单一真相源运行依赖导出，干净 staging 打包，彻底杜绝 GPU 内容泄漏至 CPU 包 (VT-05)。
- 补齐真实生命周期、快照持久性、双实例并发与内存循环测试，测试全量注册 CTest (VT-06)。
- 验证 Python 工具契约 (VT-07)。
- 恢复 AGENTS 文档闭环与唯一任务账本跟踪 (VT-08)。

**当前状态**
- 已完成：
  - VT-01 (VS-01 / F01): 删除 `src/sdk/` 目录及其所有源码和头文件；修改 `cmake/ConfigVisionRTToolSDK.cmake`，强制 `find_package(VisionRTToolSDK CONFIG REQUIRED)` 且移除所有 fallback；安装外部 SDK 并验证缺失 SDK 时 CMake 严格报错失败。
  - VT-02 (VS-02 / F02 / F07): 建立 `tools/module_bindings.yaml`；重构 `tools/generate_tool_contracts.py` 为唯一注册入口，严格使用 PyYAML 生成 `PortConstants.h`、`visiontools_*_factory.inc` 以及 `module_index.json`；接入 CMake `generate_contracts` 构建目标；生产模块消费生成工厂；编写并扩展 `tests/test_generator.py`（12 项测试）。
  - VT-03 (VS-03 / F06 / F07): 优化 `ImageCropTool` 实现轴对齐矩形裁剪通过 `image.region()` 走零拷贝只读视图（共享底层 OpenCV 内存缓冲 `u` 并保留 `rowStride`）；`AdaptiveBinaryTool` 严格校验 `constant_offset` 为有限 Float 类型（拒绝 Int 错型与 NaN/Inf）；`DetectionFilterTool` 统一校验阈值类型（拒绝非数值）；更新 `test_preprocess_contract` 验证零拷贝与严格类型校验。
  - VT-04 (VS-04 / F04): 重写 `probes/infer/CMakeLists.txt` 与 `main.cpp`，移除旧失效接口与不存在的源码引用，直接链接消费生产 `visiontools_infer` 模块与外部 `VisionRTToolSDK`，调用 `visiontoolsModuleApi()`；接入 CMake 构建并验证探针编译运行。
  - VT-05 (VS-06 / F03 / F05): CMake 导出 `resolved_dependencies.json` 作为依赖单一真相源；重写 `tools/package_tools.py`，实现干净 staging 打包，GPU 模式严格要求 `visiontools_infer.dll` 及依赖（缺失直接失败），CPU 模式彻底排除 GPU 声明 (`visionrt.dl.yolo.yaml`) 与 DLL；Python 自定义示例放入可选 `examples/` 不覆盖用户 custom；增强 `tools/build.cmd` 参数支持（`-B`、参数透传、`--no-infer` 联动打包与自动校验）；新增 `tests/test_package.py` 覆盖 9 项测试。
  - VT-06 (VS-05 / VS-08 / F08): 扩充 `tests/test_module_contract.cpp`，覆盖未知 registry (-3)、空指针防御、services 容错、release 幂等性、快照持久性、2 实例并发、100 次循环稳定性；将 Python 测试（生成器、打包、算子）全量注册至 CTest。
  - VT-07 (VS-07 / F08): 编写 `tests/test_python_tools.py` 验证 `mean.py` 缺失输入、空批、异形只读输入及 float64 rank=1 dense 输出契约，验证 `counter.py` 生命周期。
  - VT-08 (VS-08 / F09): 建立根 `README.md` 与 `docs/README.md` 索引入口与复用决策表；将 `WORKLOG.md` 从 `.gitignore` 移除并纳入版本跟踪；记录详尽验证证据。
- 未完成（客观外部资源受限）：
  - VT-04: YOLO 真实权重实测未验证（当前环境缺少 `.wts` 权重文件，探针提示 `Required: --wts FILE --engine FILE...`；已验证探针构建、接线与二进制调用，权重推理数值一致性按规范如实标记未验证）。
  - VT-07: 宿主联合加载未验证（当前独立仓库无外部宿主运行时环境；算子契约已通过独立单元测试验证）。

**验证证据**
- `python tests/test_generator.py`
  - 12/12 单元测试通过 (Ran 12 tests in 0.008s, OK)。
- `python tests/test_package.py`
  - 9/9 单元测试通过 (Ran 9 tests in 0.320s, OK，涵盖缺 DLL 失败、GPU→CPU 同目录无残留、非法路径、空格路径等)。
- `python tests/test_python_tools.py`
  - 4/4 契约测试通过 (Missing 输入、空批、异形只读输入、float64 dense 输出、counter 计数)。
- `tools\build.cmd --clean --test`
  - 完整 GPU 构建成功，CTest 6/6 全部通过：
    - `test_preprocess_contract`: Passed (0.03 sec)
    - `test_module_contract`: Passed (0.02 sec)
    - `test_infer_contract`: Passed (0.01 sec)
    - `test_generator`: Passed (0.10 sec)
    - `test_package`: Passed (0.36 sec)
    - `test_python_tools`: Passed (0.20 sec)
- `tools\build.cmd -B build-cpu --no-infer --clean --test`
  - CPU 独立构建成功，CTest 5/5 全部通过（无 InferRT/CUDA/GPU 目标）。
- `tools\build.cmd --package dist\VisionTools_GPU`
  - 成功打包 52 个文件至 `dist\VisionTools_GPU`，`tools_manifest.txt` 生成且 `verify_package` 验证通过 (variant=gpu)。
- `tools\build.cmd -B build-cpu --no-infer --package dist\VisionTools_CPU`
  - 成功打包 19 个文件至 `dist\VisionTools_CPU`，`tools_manifest.txt` 生成且 `verify_package` 验证通过 (variant=cpu)；检查确认 0 个 GPU DLL / 0 个 YOLO YAML 泄漏。
- `python docs\VisionTools_验收与改进方案\visiontools-acceptance\evidence\reproduce.py .`
  - 运行复现脚本，输出：`index_exists True`, `gpu infer_dll_exists False yolo_yaml_exists True verify False`（修复了旧版缺失 DLL 仍伪成功的问题），`cpu infer_dll_exists False yolo_yaml_exists False verify True`（修复了旧版 CPU 泄漏 GPU YAML 的问题）。

**下一步**
- 检查 Git 变更树，将所有改动分阶段或整洁地提交至 `dev` 分支并推送到远端。

### 2026-10-17 — VisionTools 分仓库全套功能交付与全量验证 (TL-01 ~ TL-09)

**目标**
- 依照方案包 (`docs/VisionRT_VisionTools_分仓库方案包.zip`) 中的 specs (`TS-01` ~ `TS-09`) 与 tickets (`TL-01` ~ `TL-09`) 完成 `VisionTools` 独立仓库的全量研发与分阶段提交。
- 实现基础设施与依赖解析 (TL-08)、InferRT CMake 接入 (TL-03)、工具声明与代码生成器 (TL-01)、预处理领域模块 (TL-02)、自适应二值化工具 (TL-07)、YOLO 工具与探针 (TL-04, TL-05)、打包脚本与分发标准 (TL-06) 以及契约测试套件 (TL-09)。

**当前状态**
- 已完成：
  - TL-08: 编写 `tools/dependencies.yaml`（SSOT 包含 OpenCV、InferRT、CUDA、TensorRT、Python、VisionRTToolSDK），`tools/build.cmd`，`tools/link_dependencies.py`，`tools/README.md`，CMake 配置模块。
  - TL-03: `cmake/ConfigInferRT.cmake` 与 `cmake/ConfigCUDA.cmake`，精准解析 InferRT 0.0.3 导出的 `InferRT::model` 目标。
  - TL-01: 整理迁移内置 7 个工具 YAML 与 Python 自定义工具示例（支持双 status 规范与类生命周期），实现 `tools/generate_tool_contracts.py` 生成 `src/generated/PortConstants.h`，编写并通过 5 项 Python 单元测试。
  - TL-02: 迁移 5 个预处理工具并实现 `visiontools_preprocess.dll` 模块导出函数 `visiontoolsModuleApi`。
  - TL-07: 实现 `AdaptiveBinaryTool`（遵循 TS-04 规范：8UC1 校验、奇数 block_size 校验、B=0 空 batch 兼容，OpenCV 像素级对齐）。
  - TL-04: 实现 `visiontools_infer.dll`，集成 `InferRT::model` 与 YOLO 后处理，对齐 TS-03 零拷贝与多通道协议。
  - TL-05: 编写 `probes/infer` 树内运行探针及参考脚本。
  - TL-06: 编写 `tools/package_tools.py`，支持 `--variant GPU/CPU`，遵循 TS-06 目录规范。
  - TL-09: 编写并通过 CTest 单元测试与打包验证套件。
  - 分阶段提交完成：6 个 git commit（全部修改为中文提交信息），不含 `WORKLOG.md` 与方案包 zip。
  - 分支管理：创建 `dev` 分支承载全部 6 次提交并推送至 `origin/dev`，`main` 分支重置恢复至 `origin/main`（初始提交）。
- 未完成：无。

**验证证据**
- `tools/build.cmd --clean --test`:
  - 完整 GPU 构建成功，CTest 3/3 通过：
    - `test_preprocess_contract`: Passed (0.07 sec)
    - `test_module_contract`: Passed (0.05 sec)
    - `test_infer_contract`: Passed (0.06 sec)
- `tools/build.cmd -B build-cpu --no-infer --test`:
  - CPU 变体构建成功，CTest 2/2 通过 (`test_preprocess_contract`, `test_module_contract`)。
- `python -m unittest tests/test_generator.py`:
  - 5/5 测试通过（合法输入、重复 ID 检测、非法 dtype 拒绝、非法 default 拒绝、键顺序无关性）。
- `python tools/package_tools.py ...`:
  - GPU 产物（`dist/VisionTools_GPU`，36 文件）与 CPU 产物（`dist/VisionTools_CPU`，18 文件）打包验证 100% 成功。
- `git push -u origin dev`:
  - `dev` 分支成功推送到远程仓库 `origin/dev`。
- `git status` / `git status --ignored`:
  - 工作树干净，`WORKLOG.md` 与 `docs/*.zip` 严格处于忽略状态，未被提交。

**下一步**
- 等待用户验收交付物或 VisionRT 主仓库完成 SDK 安装包发布后进行联调。


