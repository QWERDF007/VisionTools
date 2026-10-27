# WORKLOG

> 本文件是项目唯一的任务账本。真实日志按最新在前追加在固定示例条目之后，并固定位于其他真实日志之上；`⏳ 待你裁决` 始终固定在顶部。

## ⏳ 待你裁决

- 2026-10-18: 外部 GPU .wts 真实权重依赖与外部真实主机联调执行仍属外部环境前置阻塞，等待外部真实权重提供或部署环境具备后进行端到端实测。

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

### 2026-10-26 — 工具声明升 schema 4、条件分支双模式、读图序列化与旋转/亮度工厂注册

**目标**
- 配合 VisionRT 宿主的参数规范化与统一状态端口改造，把工具包声明升级到 schema 4，并补齐条件分支、图像读取与已实现工具的注册缺口。

**当前状态**
- 已完成：内置与自定义声明全部由 schema 3 升到 4；条件分支初始化参数新增 `condition_mode`（formula/upstream），输出新增 `on_true_status` / `on_false_status`；分支工具按条件来源派发，缺省仍走公式。
- 已完成：图像读取拆出 `ImageSequence`（批次规划）与解码缓存；`count` 上限 64；不循环时按需重扫、循环时按游标推进；解码失败拆分为用户消息与 OpenCV 技术细节，主消息不再泄漏库内部文本。
- 已完成：补齐 `visiontools.image.rotate` / `visiontools.image.brightness` 工厂注册（实现早已存在但未注册）；索引 modules=3、tools=22。
- 已完成：提交并推送 `83c6244`（声明升级）、`d6bab3d`（读图重构与工厂注册），与 `origin/refactor/abi4-writable-data` 同步。
- 未完成：无。

**验证证据**
- `ctest --test-dir build` → `100% tests passed, 0 tests failed out of 13`（`test_infer_whitebox_trace` 按设计 Skipped）。
- `module_index.json` → modules=3（`visiontools_infer` / `visiontools_metrology` / `visiontools_preprocess`），tools=22。
- 宿主校验 `visionrt_validate_tool.exe --package F:\Projects\VisionTools\dist\visiontools-deploy` → `Valid tools package`。
- 旋转/亮度执行路径由 `test_module_contract`、`test_installation_consumer` 覆盖（补齐 `center` / `canvas` / `interpolation` / `border_value` / `alignment` 等必需输入后通过）。
- 宿主安装态链路：`installed_pipeline_contract.exe F:\tmp\VisionRT\tools-package` 与 `installed_python_author_contract.exe` 均通过。

**下一步**
- 无新增动作；GPU 工具真实权重端到端联调仍受外部权重阻塞（见顶部待裁决）。

### 2026-10-25 — 依据 VisionRT_VisionTools_f6f57b03_验收报告与七工具完整改进方案 完成七工具、两个隐藏辅助工具与受限表达式引擎

**目标**
- 依据《VisionRT_VisionTools_f6f57b03_验收报告与七工具完整改进方案_Specs_Tickets_20261025》（工单 T31-VT-01 ~ T31-VT-04 与七工具目标规范、端口契约、验收矩阵），在 VisionTools 现有代码上补齐七个正式工具、两个隐藏辅助工具与合流宿主侧的声明升级。

**当前状态**
- 已完成 (T31-VT-01 预处理)：`visiontools.image.rotate`（keep/expand 画布、四角像素外边界映射与平移、`alignment_out = A_rotation * alignment_in`、`border_value` 校验）、`visiontools.image.brightness`（`convertTo` 增益/偏移、BGRA alpha 保持、ROI∩mask、`[]`=全图与 `[[]]`=该图不变语义区分、`emit_mask`）、`visiontools.image.read` 重写（mode=file/directory/list、read_mode=all/sequence、end_policy=wrap/hold/error、recursive、extensions、paths_in、limit、max_batch_bytes，空目录/空列表明确失败，输出 `current_index/total_count/end_reached`）。
- 已完成 (T31-VT-02 测量)：新增 `visiontools_metrology` 模块，`EdgeProfile`（双线性采样 + 切向平均 + Gaussian + 抛物线亚像素）、`CircleFit`（SVD 初值 + Huber 加权 GN + MAD 内点）、`CircleMeasureKernel` / `RowSpacingKernel`（配对 width/gap/pitch）及对应工具 `visiontools.measure.circle`、`visiontools.measure.row_spacing`。
- 已完成 (T31-VT-03 逻辑)：受限 AST 引擎（vendored simpleeval 1.0.3 + 白名单函数表 + `MAX_EXPRESSION_CHARS=4096` / `MAX_AST_NODES=512` / `MAX_COLLECTION_ITEMS=100000` 资源上限），`visiontools.logic.expression`（`node_model=expression`）与 `visiontools.logic.branch`（`node_model=condition`，失败返回两路 None）。
- 已完成 (T31-VT-04 辅助)：隐藏辅助工具 `visiontools.assist.circle` / `visiontools.assist.row_spacing`（`visibility=assist`，有界候选搜索 + 复用正式内核 + proposal 逆变换回参考坐标）。
- 已完成: 新增 `src/common/GeometryTransform`（Alignment 相似变换判定、clockwise 旋转、逆变换）、`RoiSupport`（Record/掩码/点内判定/编码）、`ImageSupport`、`InputValues`；`package_catalog.json` 模块增至 3 个、工具增至 22 条，全部声明升级到 `schema_version 3`。

**验证证据**
- `cmake --build build -- -j 6` → 0 错误 0 失败；`ctest` → `100% tests passed, 0 tests failed out of 13`（`test_infer_whitebox_trace` 无 GPU 轨迹按设计 Skipped）。
- 新增回归目标：`test_yolo_postprocess`（纯 CPU YOLO 后处理：IoU、几何映射、padding 过滤与上限、全 padding、部分越界、同分稳定顺序、非有限输入、参数校验）、`test_geometry_contract`（Alignment/ROI/几何契约）、`test_logic_engine`（13 项断言，覆盖白名单、资源上限与两路分支）。
- 声明契约：`tests/CMakeLists.txt` 将 `package`/`package_payload`/`python_tools`/`distribution`/`logic_engine` 统一纳入 `foreach(_test IN ITEMS ...)`，22 条声明全部通过 `visionrt_validate_tool` 校验（随 VisionRT 打包阶段 `tools package verification passed successfully` 复核）。

**下一步**
- 等待用户验收；如需继续，按后续 spec / 工单迭代。

### 2026-10-18 — 依据职责分离方案完成生成器打包与反例闭环交付 (BT-03 ~ BT-06 / BS-03 ~ BS-06)

**目标**
- 依据《VisionTools_VisionRT_职责分离_重新验收与方案_Specs_Tickets》完成 VisionTools 算子库生成器、打包流水线与测试断言整改；
- `generate_tool_contracts.py`: 强制 `--validator` 为必填参数（剔除 PATH 与 C:/dev 探测回退，缺失直接非零退出）；增加端口与 display 嵌套字段（`color_order`, `same_geometry_as_input`）白名单校验；
- `package_tools.py`: 依赖严格在 `build/<config>/resolved_dependencies.yaml` 查找（彻底移除构建根目录回退）；强制 `variant` 必填；实施 `exact tool set` 完全一致校验；校验模块 DLL 导出 `visiontoolsModuleApi` 符号；GPU 模式下 TensorRT 核心运行时排除 plugin、dispatch、lean 辅助库；
- `test_module_contract.cpp`: `evaluateRssPlateau` 严格拒绝每个窗口增长 ≥ 256KiB 的伪平台期持续增长；
- `reference.py`: 严格校验必须参数与路径，缺失时输出 `passed: false` 且非零退出（剔除静默 continue）；
- `visionrt.dl.yolo.yaml`: 更新默认 `model_name: yolov8n`（选项 `[yolov8n]`）与 `load_mode: engine`（选项 `[engine, build_wts]`）；
- `test_infer_whitebox_trace.cpp`: 在无效模型路径时退出码 1，无模型环境变量时按规范退出码 77 (`SKIP_RETURN_CODE 77`)；
- `test_installation_consumer.cpp`: 使用 SDK `parseModuleIndexFile` 动态解析并验证 CPU 包 6 个工具的完整生命周期与中文目录执行；
- 清理废弃方案包 `.zip`，维护 `WORKLOG.md`。

**当前状态**
- 已完成 (BT-03 / BS-03): 生成器强制 validator 必填与 display 字段白名单，单元测试 30/30 通过；打包器严格依赖路径、variant 必填、exact tool set 校验、DLL 导出符号核验、TensorRT core 运行时过滤，单元测试 25/25 通过；
- 已完成 (BT-04 / BS-04): `evaluateRssPlateau` 平台期断言增强；`reference.py` 严格校验输入；`visionrt.dl.yolo.yaml` 规范化默认值；`test_infer_whitebox_trace` 无模型时跳过（CTest 报告 100% tests passed, 1 skipped）；`test_installation_consumer` 动态执行 6/6 CPU 工具通过（含中文路径测试）；
- 已完成 (BT-05 / BS-05): 运行 `reproduce.py` 复核全部反例 C01~C08 与负例 N01~N10，全部符合断言通过；
- 已完成 (BT-06 / BS-06): 删除 `docs/VisionRT_db47f70_验收与下一阶段方案.zip`；维护根目录 `WORKLOG.md`；
- 未完成/未验证项: tools_infer 真实 GPU 模型推理在缺少物理 GPU 与真实模型权重时按规范隔离并报告为 `not_run`（跳过），不伪造数据。

**验证证据**
- `pytest tests/test_generator.py` → 30/30 passed；
- `pytest tests/test_package.py` → 25/25 passed；
- `tools\build.cmd python visionrt-next/visionrt-separation/evidence/reproduce.py F:\Projects\VisionTools` → 全部反例 C01~C08 与负例 N01~N10 100% 达成预期；
- `ctest --test-dir build --output-on-failure` → 8/8 tests（7 passed, 1 test_infer_whitebox_trace skipped as per specification）；
- `build\tests\test_installation_consumer.exe dist\VisionTools` → 6/6 工具动态加载并契约执行成功；
- `build\tests\test_installation_consumer.exe "dist\测试工具包"` → 中文目录动态加载与契约执行成功；
- `git status` 确认 `docs/` 下已无 `.zip` 文件。

**下一步**
- 提交并推送代码至远端分支。

### 2026-10-18 — VisionTools YAML实施验收整改与闭环交付 (YR-01 ~ YR-06 / F71 ~ F79)

**目标**
- 按照《VisionTools_YAML实施验收与下一阶段方案_ad59182》完成所有阻断与缺陷整改（F71~F79 / R71~R82）：
  - `YR-01 (F71/F78/R82)`: 跨仓统一元数据 Schema 与 SDK 依赖对齐；消除 VisionRT 宿主与 VisionTools 包的 Schema 分叉（`variant`, `modules` 映射, `tools` 映射, `runtime_files` 序列）；安装探针与安装消费者复用 SDK 的 `yaml-cpp` target，排除 Anaconda ABI 冲突。
  - `YR-02 (F76/F75/R78/R79/R80)`: 强化生成器语义校验与 YAML 纯净度；集成 `UniqueKeyLoader` 拒绝 YAML 重复键；严禁缺少 port `name`、非法 `options` 类型、未知 `options_provider`、`init_params` 与 `inputs` ID 重叠、`cpp_function` 声明包含 `init_params`；支持 `.yaml` 与 `.yml` 声明同等发现；彻底移除伪 `json.dump` 回退分支；优先调用安装 SDK 的 `visionrt_validate_tool`。
  - `YR-03 (F72/F74/F77/R71~R77)`: 彻底落实入口清空与严格闭包打包；`package_tools` 在读取元数据前执行第一行清空目标目录，消除旧包残留与标记泄漏；严格以构建索引 `build_index["tools"]` 为复制真相，空 tools 或缺失声明立即报错退出，绝不再回退扫描源码重填；严格验证 module 库名与模块名一致、工具 definition 与 tool_id 对应；强化依赖闭包校验，CPU 包必须包含 OpenCV 且严禁 GPU 相关 DLL，GPU 包必须包含 OpenCV、InferRT 与 CUDA 依赖。
  - `YR-04 (F73/F75/F78/R81)`: 升级公开探针与参考脚本；探针参数解析不全时必须返回非零退出码，严禁静默回退为合成测试；探针移除私有内部头文件 `VersionDef.h`，从唯一 `module_index.yaml` 提取库路径，严格按 ABI major、struct_bytes、create 函数指针验签；探针报告真实反映各子项状态，外部依赖或资源缺失时诚实写 `not_run`，未执行完整流程返回退出码 3；改写 `reference.py` 为纯 YAML 接口，彻底剔除 XML 与 JSON。
  - `YR-05 (F79)`: 强化内存与白盒门禁；完善 `evaluateRssPlateau`，明确识别并拒绝持续单调 +512KiB/窗口的伪平台增长并提供反例测试；拆分未初始化错误路径与真实初始化的 whitebox trace 测试。
  - `YR-06`: 清理历史验收文档与压缩包（`docs/VisionTools_YAML实施验收与下一阶段方案*`），保持文档与代码单一真相源，全量 Python 与 CTest 测试验证。

**当前状态**
- 已完成：YR-01 ~ YR-06 全部整改与验证（R71~R82 对应反例与行为全部达标；Python 52/52 全绿；CTest 8/8 全绿；CPU/GPU 双向打包与动态消费测试全部通过；清理历史文档）。
- 未完成（客观外部资源受限）：
  - 真实 GPU `.wts` 权重实测未验证（当前环境无外部 YOLO 权重文件，推理代码、编译、whitebox 测试、probe 合成测试与 package 校验已通过）。
  - 宿主联合运行未验证（当前独立仓库无真实外部 VisionRT 宿主运行时环境）。

**验证证据**
- `python -m unittest discover tests` -> 52/52 tests passed
- `tools\build.cmd --clean --no-infer --test` -> CPU 模块编译成功，CTest 6/6 (100%) 全部通过
- `tools\build.cmd --test` -> GPU/Infer 模块编译成功，CTest 8/8 (100%) 全部通过
- `tools\build.cmd --no-infer --package` -> 验证通过，成功打包 6 个 CPU 工具到 `dist\VisionTools`，验证无 GPU 残留且 OpenCV 存在
- `build\tests\test_installation_consumer.exe dist\VisionTools\bin\visiontools_preprocess.dll` -> 动态加载执行 6/6 工具无遗漏并通过
- `tools\build.cmd --package` -> 验证通过，成功打包 7 个 GPU 工具到 `dist\VisionTools`
- `build\probe_test\visiontools_infer_probe.exe --wts requested.wts` -> 退出码 1（R81 参数不全严格拒绝，不自动回退为合成测试）
- `build\probe_test\visiontools_infer_probe.exe --synthetic-test` -> 退出码 0，生成结构化 probe_report.yaml（overall_status, approved_baseline_comparison, memory_lifecycle, host_integration 均为 not_run）
- 独立验证 R71~R80 反例脚本 -> R71(空tools拒绝), R72(清空优先旧标记清除), R73(definition篡改拒绝), R74(module篡改拒绝), R75(CPU缺OpenCV拒绝), R76(GPU缺InferRT/CUDA拒绝), R77(CPU混入GPU库拒绝), R78(重复键拒绝), R79a~e(语义错误拒绝), R80(yml后缀支持) 全部通过。

**下一步**
- 等待用户提供真实 YOLO `.wts` 权重及部署环境后进行 GPU 端到端实测。

### 2026-10-18 — VisionTools YAML 统一与直接破坏性重建方案交付 (YVT-01 ~ YVT-06)

**目标**
- 依照《VisionTools YAML统一与直接重建方案 (YVT-01 ~ YVT-06 / YS-01 ~ YS-06)》完成元数据格式全量YAML统一、打包逻辑简化为直接破坏性重建、彻底移除历史包袱与临时过度设计。
- `YVT-01 (YS-01 / YS-02)`: 统一生成器与SDK合同输出：彻底废弃 `module_index.json` 与 `tools_manifest.txt`，生成轻量化 `module_index.yaml`（仅索引 variant、modules 模块库相对路径、tools 注册名对应模块及源 definition 文件、runtime_files）；删除向后兼容转换器、双写与旧格式容错；CMake 输出 `resolved_dependencies.yaml`。
- `YVT-02 (YS-03)`: 重构打包流水线为直接破坏性重建：移除原子替换重命名、临时暂存区（`.tmp_pkg_*`）、旧包备份（`.backup`）、回滚恢复及历史保留逻辑；启动时对目标目录执行直接清空重建（`shutil.rmtree(install_dir)`）；`verify_package` 严格检验 `module_index.yaml` 格式、模块库文件存在性、工具声明与索引完全一致性、以及运行时动态依赖；更新 `test_package.py` 删除旧回滚测试并新增直接破坏性重建验证。
- `YVT-03 (YS-02)`: 升级安装消费者 `test_installation_consumer`：引入 `yaml-cpp` 解析 `module_index.yaml`，彻底移除自制字符串解析与 JSON 历史兼容，遍历并严格执行所有预期工具的生命周期；兼容中文路径与分发包独立消费。
- `YVT-04 (YS-04)`: 升级公开推理探针：`probes/infer` 完全解除对 `visiontools_infer.lib` 的链接依赖，基于 `LoadLibraryW`/`GetProcAddress` 动态加载；使用 `YAML::Emitter` 输出轻量结构化 `probe_report.yaml`，彻底移除 `cv::FileStorage` XML 输出；支持 `--synthetic-test` 与纯合成自验证。
- `YVT-05 (YS-05)`: 诚实记录事实与验证边界：外部 GPU .wts 真实权重依赖与外部真实主机联调执行仍属外部环境前置阻塞，等待外部真实权重提供或部署环境具备后进行端到端实测。
- `YVT-06 (YS-06)`: 保持 WORKLOG.md 本地维护且不提交至 Git，清理历史轮次（第五轮）文档目录，按逻辑分阶段以中文 Commit 提交并推送到 `origin/dev`。

**当前状态**
- 已完成：YVT-01, YVT-02, YVT-03, YVT-04, YVT-05, YVT-06 全部代码实现、单元测试、包验证、双架构（CPU/GPU）完整构建与集成测试。
- 未完成（客观外部资源受限）：
  - 真实 GPU `.wts` 权重实测未验证（当前环境无外部 YOLO 权重文件，推理代码、编译、whitebox 测试、probe 合成测试与 package 校验已通过）。
  - 宿主联合运行未验证（当前独立仓库无真实外部 VisionRT 宿主运行时环境）。

**验证证据**
- `pytest tests` -> 45/45 tests passed (test_generator: 21, test_package: 20, test_python_tools: 4)
- `tools\build.cmd -B build-cpu --no-infer --clean --test --package dist-cpu` -> 编译成功，CTest 6/6 (100%) 全部通过，生成 module_index.yaml 并通过验证打包 6 个工具
- `build-cpu\tests\test_installation_consumer.exe dist-cpu` -> 动态加载成功，6/6 工具无遗漏执行并通过
- `build-cpu\tests\test_installation_consumer.exe "dist\测试中文路径\VisionTools"` -> 中文路径下动态加载与契约执行通过
- `tools\build.cmd -B build-gpu --infer --clean --test --package dist-gpu` -> 编译成功，CTest 8/8 (100%) 全部通过，打包 7 个工具
- `build-gpu\probes\infer\visiontools_infer_probe.exe --synthetic-test` -> 合成 YAML 报告生成与校验通过
- `build-gpu\probes\infer\visiontools_infer_probe.exe --package-dir dist-gpu --report dist-gpu/probe_report.yaml` -> 探针针对安装包生成 probe_report.yaml 通过

**下一步**
- 等待用户提供真实 YOLO `.wts` 权重和真实宿主联合运行测试环境后进行端到端全链路实测。

### 2026-10-18 — VisionTools 第五轮验收整改与下一阶段交付 (VT5-01 ~ VT5-06)

**目标**
- 依照《VisionTools 第五轮验收与下一阶段方案 (0eb7bc8)》（N01 ~ N08，AS-01 ~ AS-06，VT5-01 ~ VT5-06）完成第五轮问题修复与全面闭环验证。
- `VT5-01 (AS-01 / F51)`: 强化 SDK 声明模式与双向契约校验（端口 name 非空校验，editor 严格枚举检验，tab 严格枚举检验，step/min/max/decimals 类型一致性与非负/正有限值校验，image 交叉引用严格指向存在声明的 ImageBatch 端口；运行时 execute 失败时输出强制全部 Missing{}，成功时双向校验未知输出 key 并拦截无效输出结构，initialize 严格拦截未知配置 key 返回 -2）。
- `VT5-02 (AS-02 / F52 / F53)`: 修复示例路径判断（统一基于 tools_dir 相对路径，防止父级/同级路径误伤），包完整性强一致性校验（`module_index.json` 工具集合与 YAML 工具集合双向完全一致集合校验 `index_registries == declared_registries`，禁止子集索引隐式遗漏），严格校验 `generator_variant`、`build_infer`、`config` 构建元数据与打包目标一致。
- `VT5-03 (AS-03 / F54)`: 修复探针报告 XML 开合结构（`compareDetections` 开头添加 `file << "{" << "matches" << "["`），真实查询 CUDA 运行时/驱动版本与 GPU 设备数量/属性，SDK 版本对齐 `VersionDef.h`。
- `VT5-04 (AS-04 / F55 / F56)`: 升级索引驱动的安装消费者（动态从 `module_index.json` 解析全部预期工具并完整执行与校验；所有 6 个 CPU 工具严格经过未知配置初始化失败 -2、释放、正常初始化、执行、二次释放与安全销毁；Windows 动态加载全面升级为 `LoadLibraryW`/`SetDllDirectoryW` 支持中文路径）；`test_module_contract` 内存测试引入平台期判定断言（`REQUIRE(maxWindowDiff < 1024 * 1024)`）。
- `VT5-05 (AS-05 / F56 / F57)`: `test_infer_whitebox_trace` 调用 `executeWithTrace` 与 `tensorDescriptions` 保持白盒验证覆盖；诚实记录外部 GPU 权重与真实宿主联调环境受限事实，不伪造测试产物。
- `VT5-06 (AS-06)`: 保持 WORKLOG.md 本地维护且不提交至 Git，清理历史轮次（第四轮）文档目录，按逻辑分阶段以中文 Commit 提交并推送到 `origin/dev`。

**当前状态**
- 已完成：VT5-01, VT5-02, VT5-03, VT5-04, VT5-05, VT5-06 全部代码实现、单元测试、包验证与集成测试。
- 未完成（客观外部资源受限）：
  - 真实 GPU `.wts` 权重实测未验证（当前环境无外部 YOLO 权重文件，推理代码、编译、whitebox 测试与 package 校验已通过）。
  - 宿主联合运行未验证（当前独立仓库无真实外部 VisionRT 宿主运行时环境）。

**验证证据**
- `python docs/VisionTools_第五轮验收与下一阶段方案_0eb7bc8/VisionTools_第五轮验收/evidence/reproduce_new.py .` -> N01 ~ N08 全部通过
- `python docs/VisionTools_第五轮验收与下一阶段方案_0eb7bc8/VisionTools_第五轮验收/evidence/reproduce_previous.py .` -> E01 ~ E08 全部通过
- `python tests/test_generator.py` -> 21/21 passed (OK)
- `python tests/test_package.py` -> 19/19 passed (OK)
- `python tests/test_python_tools.py` -> 4/4 passed (OK)
- `tools\build.cmd -B build-cpu --no-infer --clean --test --package dist-cpu` -> 编译成功，CTest 6/6 (100%) 全部通过，打包与包验证通过
- `tools\build.cmd -B build-gpu --infer --clean --test --package dist-gpu` -> 编译成功，CTest 8/8 (100%) 全部通过，打包与包验证通过
- `build-cpu\tests\test_installation_consumer.exe dist-cpu` -> 动态加载通过，6/6 工具无遗漏执行并通过
- `build-cpu\tests\test_installation_consumer.exe "dist\测试中文路径\VisionTools"` -> 中文路径动态加载与契约执行通过

**下一步**
- 等待用户提供真实 YOLO `.wts` 权重和真实宿主联合运行测试环境后进行端到端全链路实测。

### 2026-10-18 — VisionTools 第四轮验收整改与下一阶段交付 (VT4-01 ~ VT4-06)

**目标**
- 依照《VisionTools 第四轮验收与下一阶段方案 (faf0738)》（E01 ~ E08，FS-01 ~ FS-06，VT4-01 ~ VT4-06）完成第四轮问题修复与全面闭环验证。
- `VT4-01 (FS-01 / E01 / E02 / E03)`: 强化 SDK schema 声明校验（端口 ID 禁止 `__` 前缀，`step <= 0` 拒绝，数值约束严格限定 Int/Float，严格禁止跨方向混用约束），生成器 formal 构建模式下 omitted `--sdk-include-dir` 严格退出 1 并拒绝隐式 fallback；在 `ValidatedToolInstance::execute` 增加输出契约全面校验。
- `VT4-02 (FS-02 / E04 / E05 / E06 / E07)`: 统一构建与发布元数据一致性（早期校验 `resolved_dependencies.json` 中的 `generator_variant` 与 `build_infer` 与目标变体匹配，不匹配拒绝覆写旧安装）；`verify_package` 严格检查 `kind` 字段、工具所属注册表存在性、Python `entry.file` 存在性；GPU 模式严格校验 TensorRT 核心运行库（匹配 `nvinfer*.dll` 并排除 dispatch/lean/plugin 等，确保不被插件库伪装通过）；样例工具 `assets/tools/custom` 从 `module_index.json` 的主工具列表剥离，与安装产物保持一致。
- `VT4-03 (FS-03 / E08)`: 公开探针消费者与白盒实现细节彻底解耦：探针仅通过公共接口验证公共输出契约（`rois`, `scores`, `classes`, `class_names`, `images`）及对比检测结果，彻底移除私有头包含、`dynamic_cast<YoloTool*>` 与 `traces.size() == batch` 假定；白盒 trace/batch 断言迁移至独立测试目标 `test_infer_whitebox_trace`。
- `VT4-04 (FS-04)`: 安装消费者与生命周期测试升级：`test_installation_consumer` 遍历并无遗漏执行所有 6 个 CPU 工具（`expectedTools == actualTools`，禁止静默跳过）；`test_module_contract` 升级为 10 轮非均匀梯度输入多轮并发比对并执行逐像素位级严格比对；生命周期测试内存判定移除 10MB 硬编码上限，改为平台期收敛检测；在分发包路径和中文目录路径下实测动态加载通过。
- `VT4-05 (FS-05)`: 诚实记录事实与验证边界，明确区分本地可测项（代码实现、编译、单元测试、安装包消费）与外部受限项（真实 GPU 权重、真实宿主环境），不把不可测伪造成已验证。
- `VT4-06 (FS-06)`: 清理历史轮次文档目录，保持 WORKLOG 本地维护且不提交至 Git，按逻辑分阶段以中文 Commit 提交并推送到 `origin/dev`。

**当前状态**
- 已完成：VT4-01, VT4-02, VT4-03, VT4-04, VT4-05, VT4-06。
- 未完成（客观外部资源受限）：
  - 真实 GPU `.wts` 权重实测未验证（当前环境无外部 YOLO 权重文件，推理代码、编译、whitebox 测试与 package 校验已通过）。
  - 宿主联合运行未验证（当前独立仓库无真实外部 VisionRT 宿主运行时环境）。

**验证证据**
- `python docs/VisionTools_第四轮验收与下一阶段方案_faf0738/VisionTools_第四轮验收/evidence/reproduce.py .`
  - E01 explicit missing SDK -> `exit: 1`
  - E02 omitted SDK fallback -> `exit: 1`
  - E03 schema reserved ID `__private` -> `accepted: false`
  - E03 schema negative step -> `accepted: false`
  - E04 missing index -> `accepted: false, old_preserved: true`
  - E04 invalid index -> `accepted: false, old_preserved: true`
  - E04 mismatch metadata -> `accepted: false, old_preserved: true`
  - E05 missing kind -> `verify: false`
  - E05 unknown registry -> `verify: false`
  - E05 missing python entry -> `verify: false`
  - E06 no tensorrt -> `verify: false`
  - E07 plugin without core -> `verify: false`
  - E08 wrapper trace conflict -> `factory_returns_wrapper: true, inner_output_returned_unvalidated: false, probe_cast_to_yolo: false, probe_requires_trace_batch: false`
- `python tests/test_generator.py` -> 21/21 tests Passed。
- `python tests/test_package.py` -> 19/19 tests Passed。
- `python tests/test_python_tools.py` -> 4/4 tests Passed。
- `tools\build.cmd -B build-cpu --no-infer --clean --test --package dist-cpu`
  - CPU 模式全量编译成功，CTest 6/6 全部通过。
  - 成功打包 19 项至 `dist-cpu`，`verify_package` 验证通过 (variant=cpu)。
- `tools\build.cmd -B build-test-infer --infer --test --package dist-gpu`
  - GPU 模式全量编译成功（含 `visiontools_infer`, `visiontools_infer_probe`, `test_infer_contract`, `test_infer_whitebox_trace`）。
  - CTest 8/8 全部通过（100% tests passed，含新 `test_infer_whitebox_trace`）。
  - 成功打包 52 项至 `dist-gpu`，`verify_package` 验证通过 (variant=gpu)。
- `build-test-infer\tests\test_installation_consumer.exe dist-gpu`
  - 动态加载 `visiontools_preprocess.dll`，6/6 工具全部执行通过（expected=6, actual=6）。
- `build-cpu\tests\test_installation_consumer.exe dist\测试中文路径\VisionTools`
  - 中文路径动态加载成功，6/6 工具全部执行通过。
- `build-cpu\tests\test_module_contract.exe`
  - 10 轮非均匀输入并发与串行逐位严格比对通过，RSS 收敛至平台期（delta = 0 字节）。

**下一步**
- 分阶段进行中文 Git 提交并推送至 `origin/dev`。

### 2026-10-17 — VisionTools 第三轮验收整改与下一阶段交付 (VT3-01 ~ VT3-07)

**目标**
- 依照《VisionTools 验收报告 (7d9d865)》（Q01 ~ Q08，RS-01 ~ RS-07，VT3-01 ~ VT3-07，P01 ~ P09）完成第三轮复验问题修复与全面闭环验证。
- `VT3-01 (RS-01 / Q01)`: 接入 SDK schema 契约实际校验，`load_sdk_enums` 严格依赖显式 `--sdk-include-dir`，去除本地兜底枚举与正则；在模块工厂包装 `ValidatedToolInstance`，在调用边界调用 `visionrt::validateToolValues(contract, ...)`，契约错误严格返回状态 `-2` 与全量 Missing 输出；分离 `spec` 子命名空间，彻底解决端口 `x` 与 `x_spec` 之间的 C++ 符号碰撞。
- `VT3-02 (RS-02 / Q02 / Q05)`: 构建元数据与配置单源，`package_tools.py` 彻底移除代码生成器调用回退与手写字典回退，构建索引缺失/损坏/变体不符严格报错拒绝；`resolved_dependencies.json` 缺失严格失败；`ConfigVisionRTToolSDK.cmake` 统一优先显式 `-D` 变量；`ConfigInferRT.cmake` 兼容直传 Config 目录与安装根目录。
- `VT3-03 (RS-03 / Q04)`: 工具包结构与运行依赖闭包，`verify_package` 拦截空 `tools: {}`、非字典条目、跨目录越界 `..`、绝对路径与包外路径；GPU 模式下补充检查 TensorRT `nvinfer*.dll` 独立物理依赖。
- `VT3-04 (RS-04 / Q03)`: 暂存预检与发布失败保护，`package_tools` 在修改目标目录前对暂存目录执行完整 `verify_package`；引入备份目录机制，支持 rename 异常回滚；确保校验失败时不触碰、不损毁既有发布包。
- `VT3-05 (RS-05 / Q06)`: 安装消费者成功口径修正与生命周期完善，`test_installation_consumer` 无参数或 DLL 不存在严格退出非零（拒绝静默返回 0）；CTest 通过显式参数传入 target DLL；增加多工具完整生命周期与契约错误状态 -2 验证；`test_module_contract` 增加双实例并发与串行输出逐像素完全相等比对，并记录详细的多点 RSS 采样序列。
- `VT3-06 (RS-06 / Q07)`: 探针消费者独立解耦，清理 Windows 下导入 `.lib` 与 `.dll` 混淆；规范条件包含私有头；客观记录真实 GPU 权重与宿主环境外部受限状态。
- `VT3-07 (RS-07 / Q08)`: 将所有负例与反例纳入正式单元测试并验证通过，WORKLOG 诚实记录事实与验证证据。

**当前状态**
- 已完成：VT3-01, VT3-02, VT3-03, VT3-04, VT3-05, VT3-06 (代码层解耦), VT3-07。
- 未完成（客观外部资源受限）：
  - 真实 GPU `.wts` 权重实测未验证（当前环境无 `.wts` 权重文件，代码解耦、编译与 CTest 测试已 100% 通过）。
  - 宿主联合运行未验证（当前独立仓库无真实 VisionRT 宿主运行时环境）。

**验证证据**
- `python docs/VisionTools_第三轮验收与下一阶段方案_7d9d865/VisionTools_第三轮验收/evidence/reproduce.py .`
  - P01: missing explicit sdk -> `exit: 1, typespec: false`
  - P02: missing binding -> `exit: 1`
  - P03: missing/corrupt/wrong_variant/broken_source -> 均被严格拦截 (`rejected: true`)
  - P04: empty / bad_entry / escaping -> `verify: false`
  - P05: missing dll -> `verify: false`
  - P06: missing tensorrt dll in GPU -> `verify: false`
  - P07: validation failure preserves previous release -> `exit: 1, previous_preserved: true`
  - P08: missing resolved_dependencies in strict mode -> `rejected: true`
  - P09: spec identifier collision -> 分离至 `in::X` 与 `in::spec::X`，命名无冲突编译通过
- `python tests/test_generator.py` -> 19/19 tests Passed (含显式 SDK 路径、非法规范化碰撞测试等)。
- `python tests/test_package.py` -> 19/19 tests Passed (含空 tools、非字典 entry、路径越界、TensorRT 缺失、发布失败旧版本保留、回滚故障注入等)。
- `python tests/test_python_tools.py` -> 4/4 tests Passed。
- `tools\build.cmd -B build-cpu --no-infer --clean --test --package`
  - CPU 独立模式编译成功，CTest 6/6 全部通过 (100% tests passed)。
  - 成功打包 19 项至 `dist/VisionTools`，`verify_package` 验证通过 (variant=cpu)。
- `tools\build.cmd -B build-gpu --clean --test --package`
  - GPU 模式编译成功，CTest 7/7 全部通过 (100% tests passed)。
  - 成功打包 52 项至 `dist/VisionTools`，`verify_package` 验证通过 (variant=gpu)。
- `build-cpu\tests\test_installation_consumer.exe dist\VisionTools`
  - 动态加载成功，测试 3 个工具（adaptive_binary, crop, detection_filter），成功调用 3 次，输出 `[PASS] test_installation_consumer passed successfully.`。
- `build-cpu\tests\test_installation_consumer.exe` (无参数)
  - 输出 `[ERROR] Missing required DLL or package path argument.` 并退出 code 1。
- `build-cpu\tests\test_installation_consumer.exe non_existent.dll` (不存在路径)
  - 输出 `[ERROR] Target path does not exist: non_existent.dll` 并退出 code 1。
- `build-cpu\tests\test_module_contract.exe`
  - 100 轮连续执行输出 `p50=7.8us, p95=10.4us`。
  - 多点 RSS 采样序列：`baseline=10723328, post_warmup=10723328, loop25=10723328, loop50=10723328, loop75=10723328, loop100=10723328, post_release=10723328, post_destroy=10723328`（0 字节漂移）。
  - 双实例并发与串行输出逐像素比对完全相等（0 差异像素）。

**下一步**
- 分阶段提交代码（WORKLOG 保持不提交），推送到 `origin/dev`。

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


