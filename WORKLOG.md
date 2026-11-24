# WORKLOG

> 本文件是项目唯一的任务账本。真实日志按最新在前追加在固定示例条目之后，并固定位于其他真实日志之上；`⏳ 待你裁决` 始终固定在顶部。

## ⏳ 待你裁决

- （已解除 2026-10-27）外部 GPU .wts 真实权重依赖：用户已提供 `F:/models/yolov8/yolov8n/yolov8n.wts`，VisionRT 侧 `gpu_tools_contract` 在 RTX 4060（sm89）双阶段端到端通过（build_wts 204.4s + engine 0.049s，5 检测与 PyTorch 参考匹配）。

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

### 2026-11-23 — 训练实现与共享契约提交

**目标**
- 将现有工具声明、业务训练及参数分组改动一并提交，并固定共享 SDK 引用。

**当前状态**
- Contracts 已统一到 df519c35c33008852ef35278124eaf85e9f28d56；两份 SDK 的暂存树完全一致，工具侧检出干净。
- 本次包含业务训练实现、30 个工具声明及发行目录、测试和文档；运行产物与 Python 缓存不纳入提交。

**验证证据**
- test_recognition_contract、test_module_contract 两项通过，见宿主 build/training-tools-final-tests.log。
- 默认工具包经宿主接受 30 个声明，安装消费者实际执行 13 个 CPU 工具，见宿主 build/default-tools-package.log；宿主 build/default-build-tests.log 的 19 项回归通过。
- 本次未修改算法或声明内容，沿用上述验证；提交前检查暂存差异和 gitlink。

**下一步**
- 工具提交身份以 Git 历史为准；宿主验收入口为 F:/Projects/VisionRT/build/bin/VisionRT.exe，尚未推送。

### 2026-11-23 — 业务实例训练与统一分组

**目标**
- 实施可选 train、统一训练参数与输出、全工具参数分组，业务版本统一 1.0.0，工具 JSON schema_version 保持 2。

**当前状态**
- 直边和颜色业务实例复用原拟合算法实现 train；移除独立训练器注册、头文件与声明。模型字节通过临时产物返回，模型路径使用共享输出，初始化通过 ToolConfiguration 接收解析后的模型引用。
- 直边训练与检测参数独立，EdgeSearchOptions 统一解析训练和运行参数。30 个工具的声明与发行目录已同步，YOLO 未声明训练能力。
- Contracts 与宿主检出的 13 个改动文件一致，ABI 为 13。完整包已安装到 F:/Projects/VisionRT/build/training/bin/tools-package；未提交或推送。

**验证证据**
- 实际四模块及消费者构建通过，见宿主 build/training-tools-delivery-build.log。
- `ctest --test-dir build -C Release -R '^(test_recognition_contract|test_module_contract)$' --output-on-failure` 两项通过，见宿主 build/training-tools-final-tests.log。
- package_tools.py --profile rtx40-build --allow-dirty --verify-runtime 经真实宿主接受 30 个声明，安装消费者实际执行全部 13 个 CPU 工具，见宿主 build/training-package-final.log。
- 宿主十四项相关验证及 SDK 隔离安装消费通过，包含真实直边与颜色训练、保存、加载、输出替换、保存失败与已有 GPU 引擎推理；入口见宿主 WORKLOG 最新条目。

**下一步**
- 运行宿主 build/training/bin/VisionRT.exe；文档索引见 docs/README.md。需要提交时先提交共享 Contracts，再同步工具和宿主引用。

### 2026-11-23 — ROI 创建、直边搜索输入与输出归属

**目标**
- 交付五个 ROI 创建工具，直边搜索区域与参考模型分离，移除六处原图透传。

**当前状态**
- 创建点、线、旋转矩形、圆和多边形通过真实模块注册与执行；无需图像即可输出结构化几何。
- 直边运行读取 inputs.6，训练调用读取同一 ROI；模型 v2 只保存实测参考边。训练和运行共用近似平行的判据。
- 颜色识别、圆测量、行间距测量、投影 ROI、缺陷过滤和 YOLO 删除原图输出，结果关联输入图像。受影响声明版本与宿主随包工程同步为 2.0.0。
- 32 工具包安装在 F:/Projects/VisionRT/build/bin/tools-package；当前提交固定共享 Contracts 引用，工具与宿主验证证据见下文。

**验证证据**
- test_recognition_contract、test_module_contract、test_preprocess_contract、test_geometry_contract 四项通过，见 F:/Projects/VisionRT/build/evidence/roi-tools-final.xml；运行搜索框和近似平行判据均先经失败用例复现。
- 真实宿主校验器接受完整包；安装消费者实际执行全部 13 个 CPU 工具，见宿主 build/roi-tools-package.log。
- 宿主 15 项定向测试全部通过，含线缆五工位、训练发布与已有 GPU 引擎推理，见宿主 build/evidence/roi-final.xml。

**下一步**
- 运行宿主 build/bin/VisionRT.exe 验收；两份 Contracts 已统一，依赖身份以父仓 gitlink 为准；实现索引见 docs/README.md。

### 2026-11-23 — 点几何接口统一

**目标**
- 适配 Contracts 统一点类型，圆测量边缘点使用相同业务几何。

**当前状态**
- RoiSupport 与 CircleTool 使用 Point 坐标列表，删除 PointSet 调用；Contracts 与宿主工作树同源，四个模块已重建为 ABI 12。
- 27 工具包已安装到 F:/Projects/VisionRT/build/bin/tools-package；尚未提交共享 SDK 与父仓改动。

**验证证据**
- 几何、识别、共享可写图像三项通过，见 build/geometry-data-tests.xml。
- package_tools.py --profile rtx40-build --allow-dirty 使用真实宿主校验器与安装消费者成功，见 build/geometry-data-package.log。
- 宿主随包示例通过，见 F:/Projects/VisionRT/build/evidence/geometry-data-integration-tests.xml；实际线缆界面截图见同目录 geometry-data.png。

**下一步**
- 后续提交先发布共享 Contracts 提交，再对齐两父仓引用；源码入口见 docs/README.md。

### 2026-11-23 — 直边训练复用业务参数

**目标**
- 删除训练与处理之间的同名参数副本，参数编辑集中到宿主下方参数区。

**当前状态**
- 直边训练调用使用已有 inputs 端口引用；移除四份重复定义，包级定位建议统一指向业务 input。
- 工作台回归与真实训练验收由 VisionRT 根任务账本统一索引。

**验证证据**
- package_tools.py 以 rtx40-build 生成 27 工具包，真实宿主校验器通过；宿主 build/image-area-tools-package.log。
- workbench_training_contract 通过真实直边训练：下方对比度 900 使训练拒绝，恢复点选建议后成功，最低完整度保留用户输入。

**下一步**
- 后续发布从本声明和发行目录表接线；安装截图与最终包身份见 VisionRT 的 build/evidence/image-area-*。

### 2026-11-22 — 语义声明、业务定位与模型训练

**目标**
- 工具独立定义算法参数；统一声明 v2，交付可被宿主复用的定位、颜色和直边训练。

**当前状态**
- 27 份声明统一语义名称，Contracts 指向 8eb2603；移除旧格式与旧工具别名。
- LocateCircle、LocateRowSpacing、LocateStraightEdge 只接收图像、搜索几何、标称区域与对齐；包级登记连接宿主。圆建议同步给出搜索半宽与平滑标准差，正式测量可直接消费。
- 颜色与直边训练器消费类型化 TrainingSet，业务节点只加载发布模型；旧训练 JSON 路径与重复读取实现已删除。
- 待父仓提交后的元数据重建及宿主最终候选安装验收；无待决定事项。

**验证证据**
- 完整 CTest 21 项中 20 通过、infer_trace_initialized 缺少环境而跳过；设置 VISIONTOOLS_TEST_YOLO_ENGINE 指向真实 yolov8n.engine 后该项通过。宿主 build/final-tools-tests.log 与 build/final-tools-gpu-trace.log 保存原始结果。
- 圆建议参数调整后 test_writable_tools_contract、test_geometry_contract、test_package、test_package_payload、test_package_profile、test_distribution 共 6 项通过，见宿主 build/final-tools-corrections.log。
- 宿主 workbench_assist_contract 以真实圆图验证正式测量与手工输入结果相同；训练、发布及重开定向验证通过。27 工具包经真实宿主校验器通过。

**下一步**
- 重建提交身份元数据，输出最终包；宿主任务账本负责安装签收证据。

### 2026-11-22 — 识色声明接线与工具包重打

**目标**
- Contracts 训练声明同步后，给 `识别颜色.json` 接上训练器与模型产物，并重打可用的工具包。

**当前状态**
- 已完成：`Contracts` 子模块更新到 `e34eaf8`（TrainingSpec、`field_id`/`parameter` 输入来源、`kTrainingManifestVersion`）。
- 已完成：`识别颜色.json` 的 `training` 增加 `call`（TrainColorModel，上下文 TrainingSet、字段 1/2）与 `artifacts`（`color-model`，`source_port 2`，目标 init 1），并补 `schema_version` 到 `颜色模型训练.json`。
- 已完成：`package_tools.py --profile rtx40-build` 重打 26 个工具，宿主校验器 `--package` 校验通过。
- 未完成：`通过直边确定对齐.json` 的同类接线，随票 10 做。

**验证证据**
- 宿主 `visionrt_validate_tool` 对两份声明返回 Valid，`--emit-normalized-json` 回写的训练块与源一致。
- `test_recognition_contract`、`test_package`、`test_module_contract` 通过；VisionRT `color_training_contract` 用本包的真实训练器产出可解码模型。

**下一步**
- 票 10：直边训练器与声明接线。

### 2026-11-22 — 颜色训练器与版本化颜色模型

**目标**
- 按 VisionRT 票 09 的工具侧：真实 `TrainColorModel` 从训练集像素与区域建模，`RecognizeColor` 只加载发布模型；训练器与业务工具共用一份模型字节格式。

**当前状态**
- 已完成：`ColorModel` 改为 `train(vector<Sample>)`，样本是 BGR 像素加 `EncodedRegion`；矩形、圆、多边形复用 `roi::regionMask`，同名标签跨图累计；不再从文件路径读训练。
- 已完成：模型字节含魔数、版本、YUV 预处理约定、正则参数、类别顺序、均值、逆协方差、对数行列式与像素计数；`encode/decode` 校验长度、有限值、类别上限与尾随字节。
- 已完成：新增隐藏工具 `TrainColorModel`（inputs 1 训练集、2 标签字段、3 区域字段；output 2 一维 uint8 模型），注册到 recognition 模块并进入 `package_catalog`。
- 已完成：`RecognizeColor` 新增初始化参数 1 为颜色模型路径（0 仍是训练来源），只 `decode`；路径缺失、无法读取与格式不兼容分别返回错误码 15/13/14，不触发重训。
- 未完成：`识别颜色.json` 的 `training.call/artifacts` 声明与 VisionRT 工作台发布验收；VisionRT 的 Contracts 训练声明尚未提交到子模块，新字段无法在本仓解析。

**验证证据**
- `ninja -C build -j 2 test_recognition_contract` 与 `ctest -R '^test_recognition_contract$'` 通过：训练集含四类、矩形、多边形、圆、第二张图的同名标签；模型类别数为 4，`decode` 后重新 `encode` 字节一致；内存模型与文件加载模型对五个工位给出相同标签、float 分数与比例；未知色、批量顺序保持不变；损坏与截断模型分别被拒绝。
- `ctest -R '^test_package'` 3/3 通过；`test_module_contract`、`test_geometry_contract` 通过。
- 未验证：VisionRT 工作台调用训练器、发布与重开（待 Contracts 同步与票 14 的示例模型）。

**下一步**
- Contracts 同步后给 `识别颜色.json` 增加 `training.call`（TrainColorModel，上下文 TrainingSet、字段 1/2）与模型产物目标 init 1；由 VisionRT 票 09 完成 A6，票 14 生成线缆示例模型。

### 2026-11-22 — 训练工具声明字段与画布约束

**目标**
- 按 VisionRT 的票 07，让训练页从声明读取训练帧端口、记录上限、字段编号与画布矩形预算，而不是由宿主按工具名猜。

**当前状态**
- 已完成：`识别颜色.json` 的 `training` 增加 `image`（inputs.1）与 `records.fields`：字段 1 颜色名称（String）、字段 2 采样区（ROI，`roi=true`，至少 1 个矩形）。
- 已完成：`通过直边确定对齐.json` 增加 `image`（inputs.1）与 `records`：`max=1`，字段 1 参考边（ROI，`roi=true`，恰 2 个矩形）。
- 未完成：`training.call`、`training.parameters` 与 `training.artifacts` 尚未声明，训练器调用归 VisionRT 票 08 与本仓 Train* 工具。

**验证证据**
- 两个 JSON 通过 VisionRT `tools_package_contract`、`tool_discovery_contract`、`validator_cli_contract` 与 `workbench_training_contract`（真实工具包加载、字段渲染、TrainingSet 交接）；本仓几何与打包测试未受影响。
- 未验证：安装包内的训练器端到端调用（待票 08/14）。

**下一步**
- 票 08 声明并接上 `TrainColorModel` / `TrainStraightBorders` 的输入输出；票 09/10/11 补算法与助手。

### 2026-11-21 — 行间距助手追加意图与声明迁移

**目标**
- 按 VisionRT 的票 06，让行间距一次 Ctrl+单击提交完整矩形组，并发布匹配 `authoring.easy_touch` 的 `PickRowSpacing` 与 `MeasureRowSpacing`。

**当前状态**
- 已完成：`行间距智能点选.json` 增加 inputs.7；`行间距测量.json` 删除 `interaction`，改 `authoring.version=1`，writes 按 outputs.3/6/7/8 写 inputs.2/6/7/8。
- 已完成：`assistRowSpacing` 在 append 时保留矩形种子并追加新带；新增 `toCurrentRectangle` 做参考坐标到当前帧的变换；非矩形种子跳过。
- 已完成：VisionRT 宿主同步收口追加语义（append 只写 ROI，共享参数不覆盖），该改动在宿主仓，不在本仓。
- 未完成：发布包未重新打包；直边助手（票 10/11）尚未迁移。

**验证证据**
- `ctest -R '^test_geometry_contract$'` 通过，新增 `assistRowAppend` 覆盖 append 与 replace。
- VisionRT `workbench_assist_contract` 在真实条带图上通过四个目标写入与追加保留参数；10 个相关合同全通过。

**下一步**
- 票 10/11 迁移直边助手时复用同一 append 端口与完整组契约。

### 2026-11-21 — 圆助手追加意图与圆测量声明迁移

**目标**
- 按 VisionRT 的票 05，让圆测量一次 Ctrl+单击提交完整圆组：助手接收追加意图并返回保留组加新圆，产线声明从 `interaction` 迁到 `authoring.easy_touch`。

**当前状态**
- 已完成：`AssistCircleTool` 新增可选 inputs.7（追加意图）；`assistCircle` 在 append 时跳过种子作为标称几何，改为光标搜索，并把种子按参考坐标保留、按当前坐标预览，新圆追加在末尾。
- 已完成：`圆智能点选.json` 增加 inputs.7；`圆测量.json` 删除 `interaction`，改 `authoring.version=1`，call 映射 inputs.1/2/3 与 cursor/phase/append，writes 按 outputs.3/7/6 写 inputs.2/5/9。
- 已完成：`test_package` 的重复字段断言改为接受校验器的实际措辞（Contracts 声明下沉后由“重复成员”变为“对象包含重复字段”）。
- 未完成：行间距助手（票 06）尚未声明 inputs.7；发布包未重新打包。

**验证证据**
- `ctest -R '^test_geometry_contract$'` 通过，新增 `assistAppend` 覆盖 append 与 replace 两种语义。
- `ctest -R '^(test_package|test_package_payload|test_package_profile)$'` 3/3 通过。
- `cmake --build build --target visiontools_metrology` 通过；产物与两份声明同步到 VisionRT 的 `build/bin/tools-package`（`VISIONRT_TEST_TOOLS_PACKAGE` 实际指向处）与 `build/tools-package` 后，VisionRT 的 `workbench_assist_contract` 在真实工作台与真实圆助手上通过。只同步 `build/tools-package` 时测试仍读旧声明，属于假绿。

**下一步**
- 票 06 复用同一 append 端口实现行间距追加；票 14 收口时删除旧 `interaction` 与 `training.kind`。

### 2026-11-20 — 汇总数字协议与工具包交付

**目标**
- 提交并推送数字声明、生成式SDK、逻辑工具、训练读取和相关功能测试。

**当前状态**
- 已完成：Contracts e0af8c6 已推送 origin/main，与 VisionRT 使用同一提交。
- 本条与本轮工具实现、作者声明、打包脚本、文档和测试一并提交，发布分支为origin/dev。

**验证证据**
- 操作前dev与远程同步；所有待提交源文件通过 git diff --check。
- 原生、Python及发行8项通过，SDK依赖调整后4项复验通过；25工具rtx40-runtime包通过真实宿主校验，具体日志见前一条记录。
- 两仓 Contracts 的准备索引树与已发布提交一致，通过无强制的切换同步子模块引用。

**下一步**
- 发布状态通过 git status -sb、git ls-remote 与 Git 历史核对；无需重复执行已通过的测试。

### 2026-11-20 — 数字枚举声明与生成式 Python SDK

**目标**
- 配合 VisionRT 样式与协议重构，统一 C++、QML、Python 使用的有限状态及声明格式。

**当前状态**
- 已完成：作者 JSON 改用数字枚举和结构化 TypeSpec；训练资产读取使用共享版本与 TrainingKind；逻辑工具按共享枚举解析类型和条件输入模式。
- 已完成：SDK 枚举从 Contracts 头文件导出，构建与打包复用同一生成器；构建侧复制 SDK 源文件时跟踪输入依赖。删除声明常量、静态源码等重复断言，保留真实 Python、原生模块和发行流程测试。
- 已完成：rtx40-runtime 包含25个工具，通过真实 VisionRT 校验器；产物位于 F:/tmp/VisionTools-ui、F:/tmp/VisionTools-ui.zip。未提交、未推送、未部署到用户安装目录。

**验证证据**
- `cmake --build build` 的受影响原生目标构建通过，日志：F:/tmp/visiontools-final-build.log。
- 原生几何、识别、模块注册及 Python、逻辑引擎、资源收集、发行事务8项全部通过，日志：F:/tmp/visiontools-verified-results.log。
- SDK 输入依赖调整后重新配置，Python、逻辑引擎、声明发行及资源收集4项通过，日志：F:/tmp/visiontools-sdk-verified-results.log。
- `python tools/package_tools.py --build-dir build --install-dir F:/tmp/VisionTools-ui --profile rtx40-runtime --validator F:/Projects/VisionRT/build/bin/visionrt_validate_tool.exe --allow-dirty` → 25个工具通过，日志：F:/tmp/visiontools-delivery-package.log。

**下一步**
- 当前实现与包验证完成；如继续提交工作，先提交 Contracts 的共享变更，再更新两仓 gitlink。宿主状态见 F:/Projects/VisionRT/WORKLOG.md。

### 2026-11-16 — 汇总识别工具提交验收

**目标**
- 将直边对齐、颜色识别、EasyTouch 直边助手及共享边缘内核改动提交并推送至 VisionTools dev。

**当前状态**
- 已完成：核对识别模块、三份声明、发行目录表、nlohmann JSON 单头文件依赖及回归测试；Contracts 子模块无未提交改动。
- 具体算法与接口以识别模块及前述两条任务记录为准，本轮仅汇总提交，不扩大实现范围。

**验证证据**
- `cmake --build build --target visiontools_recognition test_recognition_contract test_geometry_contract test_module_contract test_output_ownership_contract` → 构建通过。
- `ctest --test-dir build -R '^(test_recognition_contract|test_geometry_contract|test_module_contract|test_output_ownership_contract)$' --output-on-failure` → 4/4 通过。
- `git diff --check` → 通过；提交前 `HEAD...origin/dev` 为 `0 0`。
- `tools/build.cmd` 本机无法正确解析；使用 MSVC 14.29.30133 与 Windows SDK 10.0.19041.0 环境直接运行 CMake 完成上述构建。

**下一步**
- 以 `git status -sb` 与 `git ls-remote origin refs/heads/dev` 核对远程提交同步状态。

### 2026-11-16 — 直边智能点选助手

**目标**
- 让对齐工具的训练页复用工作台的 EasyTouch：在参考图上点一下边缘就得到一条参考边与搜索框，而不是由宿主自造"画矩形"开关。

**当前状态**
- 已完成：`PickStraightEdge`（`ui.visibility = assist`，声明在 `处理/识别/直边智能点选.json`），复用计量助手的 cursor/phase/seeds/alignment/search_extent 协议端口与 preview/proposal 输出。
- 已完成：`StraightEdgeProbe` 先用结构张量在光标窗口估计边缘方向（要求相干性 ≥ 0.2），再用已有的 `locateSegment` 在该方向的带状搜索框内拟合直线；返回值是"长边沿边缘、短边为搜索范围"的旋转矩形，正好是训练资产里一条 reference edge 的几何。
- 已完成：回归目录 `package_catalog.json` 增加该工具（发行 25 个工具），`test_recognition_contract` 增加三例：水平边缘给出水平带、垂直边缘给出 90° 带、空白处不产出候选。

**验证证据**
- `cmake --build build --target visiontools_recognition test_recognition_contract` → 通过。
- `tests/test_recognition_contract.exe` → 通过（含新增 `straightEdgeProbe` 三例）。
- `python tools/package_tools.py --profile rtx40-build` → 25 个工具打包成功，模块表含 `visiontools_recognition`。

**下一步**
- 尚无：颜色训练仍是手工框选矩形，若以后也要点选，可另加 `PickColorRegion` 助手。

### 2026-11-16 — 新增识别领域：直边对齐与颜色识别

**目标**
- 按 `VisionRT/docs/VisionRT_颜色识别_修订方案` 的 P0 落地两个 CPU 工具：通过直边确定对齐、识别颜色；训练数据作为唯一源，不新增 Contracts 训练契约与独立模型文件。

**当前状态**
- 已完成：`metrology/EdgeProfile` 上移到 `common`（命名空间 `visiontools::edge`），计量工具改为引用共享内核，直边工具复用同一套双线性采样、梯度与极性检测。
- 已完成：新增 `src/recognition` 模块与错误域 `kRecognition`；`TrainingAssets` 解析两种训练资产（`straight_edges`、`color_samples`）；`AlignStraightBordersTool` 输出参考到当前的刚体变换、找到的边与搜索区域；`RecognizeColorTool` 在 U/V 上按类高斯分类，输出颜色、分数、分类像素比例与中心坐标，门限不足输出 `unknown`。
- 已完成：两份工具声明（`处理/识别/`）、`package_catalog` 模块与工具归属、`visiontools_recognition` 目标、`test_recognition_contract`。
- 已完成：`识别颜色.json` 增加 `training` 块（`color_samples`，资产参数 0），由 VisionRT 宿主按工具声明显示训练入口；直边对齐工具暂不声明训练，避免出现无实现的 Tab。
- 未完成：VisionRT 的训练能力声明、处理/训练 Tab 与训练会话；示例工程与真实 `cable_11` 回归；11 张原始样本的逐图验收。
- 设计取舍：训练资产只保存图片路径与几何，工具在 `initialize` 重建统计模型，因此应用训练数据不会留下第二份派生模型；相对路径沿用产品规则，以主程序目录为基准。

**验证证据**
- `cmake --build build --target visiontools_recognition` → `vst_recognition.dll` 链接成功。
- `cmake --build build --target test_recognition_contract test_geometry_contract test_module_contract test_output_ownership_contract` → 四个目标编译通过。
- `tests/test_recognition_contract.exe` → 通过：参考图输出近似单位变换（平移 0.5 px 像素中心量化内）、已知旋转+平移姿态误差小于 0.75 px、极性不符返回 `EdgeNotFound` 且三个业务端口为 Missing、五圆颜色索引为 purple/red/blue/yellow/purple、未训练绿色为 `unknown`、双图批量按图保持索引。
- `tests/test_geometry_contract.exe`、`tests/test_module_contract.exe`、`tests/test_output_ownership_contract.exe` → 全部通过，边缘内核上移未改变计量行为。
- `tools/package_catalog.json` 与全部声明的模块归属、端口编号唯一性用脚本核对通过（24 工具、4 模块）。
- 未验证：`visionrt_validate_tool.exe` 对当前声明报 `name_en` 未知字段，但现有 `圆测量.json` 同样报错，判断为该二进制与当前 Contracts 不一致；宿主实际解析未在本轮验证。

**下一步**
- 在 VisionRT 增加工具声明的可选 `training` 描述与训练会话，接处理/训练文字 Tab、训练记录列表与应用训练数据；随后组装 `examples/03_综合示例/02_识别线缆颜色` 并用真实样本回归。

### 2026-11-14 — 调整亮度始终发布已声明的应用区域

**目标**
- 修掉「输出应用区域=false 时工具只发布端口 2」与声明冲突：宿主按声明校验端口 3，默认参数下必然报「工具未发布输出端口：端口 3」。

**当前状态**
- 已完成：`调整亮度.json` 移除输入端口 7「输出应用区域」，`next_port_id.inputs` 收敛到 7；`BrightnessTool`/`BrightnessParameters` 删除该开关，成功执行时始终发布端口 3 的应用区域；`test_geometry_contract` 的“未请求时保持缺失”断言改为“声明即输出”，另两处测试去掉该输入。
- 未完成：按需输出若仍要保留，需要在 Contracts 声明里引入可选出端口（宿主校验按声明要求成功即全发布，当前无此概念）；本轮不引入该机制。

**验证证据**
- `ctest -R "test_module_contract|test_geometry_contract|test_upstream_status_contract|test_output_ownership_contract"` → 4/4 Passed。
- `python tools/package_tools.py --profile rtx40-build --install-dir dist/visiontools-deploy --allow-dirty` → 22 个工具打包完成，部署目录与 ZIP 内 `调整亮度.json` 已无端口 7。
- VisionRT `installed_pipeline_contract` 与 `example_pipelines_contract` 在更新后的部署包上通过，确认函数注册路径执行正常。

**下一步**
- 无。

### 2026-11-04 — R45：工具声明与实现迁移到数字 PortId 与 schema 6

**目标**
- 按 R45 把 22 份工具声明、C++ 工具读写、Python SDK 与测试从字符串端口名切到数字 PortId，声明升到 schema 6，并同步 Contracts ABI 9。

**当前状态**
- 已完成声明：22 份 `assets/tools/**` 全部 schema 6，业务端口写 `id` 并新增 `next_port_id`；输出 0/1 为状态码与状态信息、业务输出从 2 起，输入 0 为上游状态。
- 已完成 C++ 侧：`InputValues`/`ToolOutputs`/`UpstreamStatus` 及各 preprocess/metrology/infer 工具按 `PortId` 取参与写输出，不再按名称查找。
- 已完成 Python 侧：author_sdk 的 `UPSTREAM_STATUS_PORT = 0`、`RESERVED_PORTS = (0, 1)`，输出字典使用整数键；内置与示例工具同步，去掉名称键与隐式字符串强转。
- 已完成测试与探针：断言头、各工具测试、逻辑引擎与安装消费测试改用数字端口。
- 已完成子模块：`3rdparty/VisionRTContracts` 指向 ABI 9（139c09b，已推送）。
- 未完成：本轮未重跑全量测试矩阵；GPU/trace 按既有 skip 规则，未在本轮执行。

**验证证据**
- `dist/visiontools-deploy.zip` 由本工作树构建（`build_info` commit `e69a2ed`，dirty=true），经宿主 `visionrt_validate_tool.exe` 校验通过；部署到 `F:\tmp\VisionRT` 后注册 22 个工具、3 个模块（见 VisionRT 部署日志）。
- 22 份 `assets/tools/**` 声明扫描结果均为 schema 6。
- `build/Testing/Temporary/LastTest.log`：2026-11-03 17:15 `test_geometry_contract` 通过；其余组本轮未重跑。

**下一步**
- 下次改动工具实现时按 backend/bridge 分组补跑受影响测试；GPU 项按所选 profile 执行。

### 2026-11-02 — R43：测试按需构建与断言头改名

**目标**
- 按 T43-06 把测试可执行移出默认构建，按 T43-09 改掉易误读的 `result_view.h`。

**当前状态**
- `visiontools_add_test` 统一 `EXCLUDE_FROM_ALL`，新增 `visiontools_tests` 目标覆盖全部测试可执行（含多场景 runner）；`tools/build.cmd --test` 先构建该目标；README 与 `tools/README` 更新。
- `tests/result_view.h → tests/ToolResultAssertions.h`，6 处测试与 `probes/infer/main.cpp` 的 include 同步。
- 默认构建只产出产品与探针；测试需显式 `--target visiontools_tests`（或 `--test`）。

**验证证据**
- 删除 `image_read_cache_test.exe` 后默认构建未重建；`visiontools_tests` 构建成功（13 个测试可执行）。
- 代表性 6 项测试通过（缓存快速/基准、追踪两场景、几何、模块契约），0.96 s。

**下一步**
- 打包与发布流程继续用默认构建；测试入口统一走 `visiontools_tests`。

### 2026-11-02 — R43：缓存/追踪测试分层与逻辑 ndarray 输入

**目标**
- 按 VisionRT `docs/R43` 的 T43-02、T43-13 拆分测试成本并接通逻辑工具的 ndarray 输入。

**当前状态**
- 已完成 T43-02：`python/builtin/logic/common/engine.py` 在唯一输入归一入口接受 `numpy.ndarray`：先校验 rank/dtype/容器数/累计预算，再一次 `tolist()` 并原地校验；list 路径语义不变。`tests/test_logic_engine.py` 增 7 项（list/ndarray 等价、rank2 int32、分支、dtype/rank 拒绝、预算、`shape=(n,0)` 容器上限、NaN/字符串）。
- 已完成 T43-13：`test_read_cache_contract` 拆为 `image_read_cache_test`（三张小图、LRU/保护/reload/失败回滚/第二实例/唯一临时目录）与 `image_read_cache_benchmark`（`--perf`，尺寸/图数/轮次/证据目录显式传入，`LABELS perf`）。
- 已完成 T43-13：`test_infer_whitebox_trace` → `infer_trace_test`，按 `--case geometry|uninitialized|initialized-trace` 注册三个条目；`geometry`/`uninitialized` 归 `smoke`，`initialized` 归 `gpu` 并保留 `SKIP_RETURN_CODE 77`。
- `cmake/AddTest.cmake` 的 `visiontools_add_test` 增加 `NO_TEST`/`TEST_NAME`/`LABELS`/`SKIP_RETURN_CODE`，供一个二进制注册多个场景，不引入新框架。

**验证证据**
- `ctest -R "^(image_read_cache_test|image_read_cache_benchmark|infer_trace_geometry|infer_trace_uninitialized|test_geometry_contract)$"` → **5/5 通过**（0.89 s）；快速缓存项 0.20 s、基准 0.66 s（12×1600×900、120 帧，证据 `build/evidence/read_cache_evidence.json`）。
- `ctest -R "^infer_trace_initialized$"` → Skipped（77，无引擎/权重）。
- 反向验证：让引擎拒绝 ndarray → `NumpyInputs` 5 失败 2 错误（后续已恢复正常实现）。
- 未验证：GPU 实机 trace（缺 fixture 时按 skip 记录）、全量 ctest。

**下一步**
- 候选发布时按需运行 GPU/trace 与完整包测试；其余命名建议随实际修改局部迁移。

### 2026-10-30 — U05：ROI 消费方迁移到 Contracts 语义视图

**目标**
- 配合 VisionRT 统一方案 S01：Tools 不再维护自己的 ROI 解码 DTO 和魔数索引，改为直接使用 `visionrt::EncodedRegion` 与命名视图；ROI 存储与端口格式不变。

**当前状态**
- 已完成：
  - `roi::Record` 改为 `visionrt::EncodedRegion` 别名，删除 `kind + values` DTO 与 `encoded()` 二次编码；`groups()` 在入口用 `RegionView` 一次校验（空记录、非法 kind/长度/非有限值仍报错），下游直接持有已校验编码的副本。
  - `contains/regionMask` 改用 `RectangleView/CircleView/PolygonView`；裁剪、ROI 投影、检测过滤、YOLO 检测输出、圆测量、行间距测量、Assist 种子/转换改用视图与 `make*` 构造。
  - `3rdparty/VisionRTContracts` 同步 Contracts 的 ROI 视图与 `RegionEditor::translate/setVertex`。
- 未完成：无（本仓 U05 范围）。

**验证证据**
- `cmake --build build --target visiontools_preprocess visiontools_metrology visiontools_infer` 通过。
- `ctest -R "^(test_writable_tools_contract|test_upstream_status_contract|test_preprocess_contract|test_read_cache_contract|test_output_ownership_contract|test_geometry_contract|test_module_contract|test_installation_consumer|test_yolo_postprocess|test_infer_contract)$"` → 10/10 通过；`test_infer_whitebox_trace` 无 GPU 跳过。
- ROI 解码搜索：`src` 下不再出现 `Record.values/kind` 或 `record[...]` 形式几何解码。

**下一步**
- 随 VisionRT U11/U12 的 Array 协议批次同步 Python 作者 SDK。

### 2026-10-28 — 亮度可空输入、出包集合完整性与 Python 作者 SDK 单源

**目标**
- 执行三仓审核包 T39-02（亮度可空图像）、T39-05（profile 最终集合与失败事务）、T39-07（Python 作者 SDK 单源）。

**当前状态**
- 已完成（亮度，[BrightnessTool.cpp](src/preprocess/BrightnessTool.cpp)、[BrightnessParameters.h](src/preprocess/BrightnessParameters.h)）：`images` 改为可空读取，先执行工具自身的上游状态策略、再在确实需要图像时报 1001；声明 `required:false` 与实现顺序一致。
- 已完成（出包，[package_tools.py](tools/package_tools.py)、[package_profile.py](tools/package_profile.py)）：被 profile 排除的模块，其库文件与依赖工具同批剔除并在 `dependencies.json` 写明原因；运行时安装目录从创建起纳入 `finally`；`runtime_files` 只记录真正落地的文件；`engine_build` profile 缺目标架构 builder 资源时直接失败；新增 `inferrt_*` 分类，CPU 包不再夹带推理运行时。
- 已完成（SDK，[python/author_sdk/visiontools_sdk](python/author_sdk/visiontools_sdk/__init__.py)、[package_payload.py](tools/package_payload.py)、四个示例）：仓库只保留一份 SDK，出包时注入每个 Python 工具包；示例改用普通相对导入，删除 `sys.path` 兜底与四份手写副本。
- 已完成（测试）：`test_upstream_status_contract` 增加“缺图 + 非零上游仍跳过”；`test_package_profile` 增加缺 SM 资源拒绝与 runtime profile 不要求 builder；`test_package` 增加安装失败不留 `.runtime-*` 与 CPU profile 剔除被排除模块库；`test_python_tools` 改为按宿主形态（包根 + 子模块）加载，SDK 从唯一源解析。
- 未验证：GPU 设备侧未复跑（本次不涉及推理代码路径）；`dist/visiontools-deploy` 已按 rtx40-build 重新出包，未见运行态回归证据。

**验证证据**
- `ctest --test-dir build -R "test_python_tools|test_package|test_logic_engine|test_installation_consumer"` → **6/6 通过**；
- `python tests/test_package.py`（真实校验器）→ **46/46 通过**；`python -m unittest tests.test_package_profile` → **15/15 通过**；
- `ctest -R "test_upstream_status_contract|test_preprocess_contract"` → **2/2 通过**；
- 真实 CPU 出包 `package_tools.py --profile cpu` → 21 工具、2 模块，`bin/` 仅 opencv + vst_image + vst_metrology，`vst_infer.dll`/`inferrt_*`/`nvinfer_*`/`cudart*` 均被排除并记入报告；
- 真实 rtx40-build 出包 → 22 工具、3 模块，保留 `nvinfer_builder_resource_sm89_10.dll`；
- `dist/visiontools-deploy` 重新出包后每个 Python 工具包内含 `visiontools_sdk/`；VisionRT 侧 `python_contract`、`cpu_tools_contract`、`installed_python_author_contract`、`examples_contract` 4/4 通过。

**下一步**
- 若继续 T39-12：在真实安装目录复跑七工具与 EasyTouch 交互回归；
- 出包夹具已可从 `--profile cpu` 直接生成，`dist/cpu-fixture` 保留为 VisionRT 测试包来源。

### 2026-10-28 — 工具错误目录与 Python 信封迁移（配合 VisionRT Contracts ABI 6）

**目标**
- 对齐 `VisionRT/docs/VisionRT_错误处理重构_最终版/`：工具只返回一个 `status`，业务码按固定域分配，Python 工具改用 `{status, outputs}` 信封；不再向 `outputs` 写 `__status_code` / `__status_message`。

**当前状态**
- 已完成（错误目录）：新增 [ToolErrorDomains.h](src/common/errors/ToolErrorDomains.h) 作为唯一域表（1001–1999 公共、101xxx 读图、102xxx 裁剪、103xxx 几何、104xxx 计量、301xxx 推理、401/402 Python 逻辑）；[ToolError.h](src/common/errors/ToolError.h) 只留 `ToolFailure` / `status()` / `fail()` 等词汇，`ToolError.cpp` 负责目录文本。
- 已完成（各工具）：`ReadImageTool`、`ImageCropTool`、`DetectionFilterTool`、`ImagePreprocessTool`、`RotateImageTool`、`RoiProjectionTool`、`BrightnessTool`、`CircleTool`、`RowSpacingTool`、`AssistTools`、`AdaptiveBinaryTool`、`YoloModel` 全部改为返回式错误；读图码严格按方案 101001 路径、101002 没有可用图像、101003 文件读取失败、101004 解码失败、101005 超出大小限制；YOLO 恢复 301001–301005 语义。
- 已完成（消息用词）：Contracts 增加 `typeLabel()`，参数读取的失败消息改用中文类型名（期望整数/浮点数/…），测试同步。
- 已完成（Python）：`visiontools_sdk` 增加 `status()/result()/success()/failure()/skipped()`，`OutputParametersBase.pack()` 返回信封、`status()` 返回生命周期状态；`builtin/logic/common/status.py` 与 expression/branch 工具沿用同一信封；四个 custom 示例与 builtin 示例全部迁移，`errors.py` 定义域内业务码。
- 未完成：`tests/test_distribution.py` 需外部 `VISIONTOOLS_VALIDATOR` 才能跑（环境依赖，非代码问题）。

**验证证据**
- `ctest --test-dir build --output-on-failure` → **17/17 通过**（含 preprocess、geometry、read cache、output ownership、module、installation consumer、infer、package 与 python 用例）；
- `python -m unittest tests.test_python_tools tests.test_logic_engine` → 32/32 通过；
- `cmake --build build` → 退出码 0，`vst_image.dll` / `vst_metrology.dll` / `vst_infer.dll` 重新产出并同步到 `dist/visiontools-deploy` 与 VisionRT 的 `build/tools-package`、`build/bin/tools-package`；
- VisionRT 侧 `ctest -L backend` 37/37 通过（真实工具包链路）。

**下一步**
- 无。工具仓库侧改动已全部通过自身契约测试。

### 2026-10-28 — 出包：构建元数据单一来源，发行包带上真实构建时刻

**目标**
- 让 `dist/visiontools-deploy.zip` 的 `build_info.json` 与真实构建一致，避免关于弹窗显示旧时间或“构建信息不可用”。

**当前状态**
- 已修复：`tools/package_tools.py` 原来优先读 `<build-dir>/<config>/build_info.json`，那是旧流程 `file(GENERATE)` 留下的文件（commit `5a69d2b`、`build_time_utc` 格式），会把新的 `build/build_info.json` 挡住。现在只认 `<build-dir>/build_info.json`，且要求存在正整数 `built_at_epoch`，旧格式直接拒绝出包。
- 已清理：删除 `build/Release/build_info.json` 这个陈旧副本（构建产物，可再生）。
- 已完成：按 `--profile rtx40-build` 重新出包，22 工具，目录与 ZIP 内的 `build_info.json` 一致（commit `077706f`，`built_at_epoch` 1793148779）。

**验证证据**
- `python test_package.py`（设 `VISIONTOOLS_VALIDATOR` 与 OpenCV bin 在 PATH）→ 45 个用例全部通过，含新增“旧格式被拒绝”和“配置子目录旧文件不得覆盖”两个用例；
- `python tools/package_tools.py --build-dir build --install-dir dist/visiontools-deploy --profile rtx40-build` → `已打包 22 个工具`，退出码 0；
- `dist/visiontools-deploy/build_info.json` 与 ZIP 内同名文件均为 `commit 077706f… / built_at_epoch 1793148779`。

**后续（同日完成）**
- `dist/` 下 15 项出包备份（5 个目录、5 个 ZIP、5 个 module_index 副本，合计约 13.9 GB）已清理；`visiontools-deploy`（22 工具 / 3 模块）与 `visiontools-deploy.zip` 保留，清理后重新校验模块索引与文件数正常。

### 2026-10-28 — VT-02/03/04：读图缓存实测、输出所有权收敛、missing 端口 helper 归并

**目标**
- VT-02：用实测替代“512 MiB 缓存”的推断，量化预算、驻留、命中率、冷热延迟与多实例影响；
- VT-03：先证明 `RotateImageTool`/`BrightnessTool` 的输出不共享输入缓冲，再去掉多余的 owner 保留链；
- VT-04：把长端口列表的 `missingOutputs()` 手写循环收敛到公共 helper。

**当前状态**
- 已完成（VT-02）：新增 `tests/test_read_cache_contract.cpp`。`DecodedImageCache` 预算强制（8 MiB 预算下 3 MiB 图像、LRU 淘汰、`protect` 保护、超预算图像拒绝且不改变已驻留字节）；工具级 12 张 1600×900 图冷/热两轮、120 帧循环、双实例隔离。`ReadImageTool::kCacheBudgetBytes` 取代头文件里的魔法数，测试与实现共用同一常量。
- 已完成（VT-03）：新增 `tests/test_output_ownership_contract.cpp`，断言输出每个平面与输入 `data`/`ptr(0)` 均不相同，并在丢弃输入后校验输入 owner 令牌已释放、输出仍可读。两工具先跑红（保留 owner），去掉 `DataRef` 的 keepAlive 后转绿；临时把 Brightness 改回保留 owner 可再次复现失败，证明门槛有效。
- 已完成（VT-04）：新增 `common/ToolOutputs.h` 的 `missingPorts({...})`，`CircleTool`、`RowSpacingTool`、`DetectionFilterTool` 三处长端口列表改用它；一两个端口的工具保留内联写法。
- 已完成（VT-01 复核）：完整构建后 `build/build_info.json` 的 `built_at_epoch` 由 1793082712 前进到 1793148757，确认“取产物 mtime”在真实重链时才更新，不在无改动构建时伪造时间。
- 未做：VT-02 未在 GPU 上采样（缓存是主机内存，与显存无关）；未在干净机器验证打包产物的运行态依赖。

**验证证据**
- `cmake --build build` → 退出码 0，无 `error C/error LNK`；`build_info.json` 时间戳随真实重链前进；
- `ctest -R "^(test_preprocess_contract|test_geometry_contract|test_upstream_status_contract|test_writable_tools_contract|test_module_contract|test_read_cache_contract|test_output_ownership_contract)$"` → **7/7 通过（1.18 s）**；
- VT-02 实测（`build/evidence/read_cache_evidence.json`）：预算 536870912 B；12 张 4.32 MB 解码帧，冷 113.6 ms / 10 帧，热 0.19 ms / 12 帧，120 帧循环 1.66 ms；工作集 12.3 → 62.7 MiB（+50.4，与 51.8 MB 解码帧相符），循环后不再增长；第二个实例冷启动重新解码 12 帧、热轮 0 次；
- VT-03 门槛自检：恢复 owner 保留时测试报 `the output retained the input owner although it allocated its own pixels: brightness must return pixels it owns`。

**下一步**
- 若继续：在干净机器验证 `dist` 包安装/加载，并补一次 GPU 路径的读图+推理组合采样（与本票的主机内存结论分开记录）。

### 2026-10-27 — T36-18 参数 SDK 迁移（ReadImageTool 参数子类、Python visiontools_sdk）

**目标**
- 完成 T36-18：ReadImageTool 保持类并复用参数子类；Python 函数（image_add）与计数器类（counter）接入统一参数 SDK 并随包安装；在真实安装目录复验。

**当前状态**
- 已完成：新增 `src/preprocess/ReadImageParameters.h`（`ReadImageInputs`/`ReadImageOutputs`，复用 Contracts 的 `InputParametersBase`/`OutputParametersBase`）；`ReadImageTool.cpp` 的输入解析、commit、同批次快速路径与全部异常路径改走参数子类，失败输出为五端口显式 Missing，upstream 类型错误保持原传播语义，缓存/事务行为不变。
- 已完成：新增 Python 作者 SDK `visiontools_sdk`（InputReader/OutputWriter + InputParametersBase/OutputParametersBase，状态键归基类、图像与列表零拷贝），作为包内子目录随 image_add 与 call_counter 安装；直接以文件加载（测试）时回退同目录导入。
- 已完成：`image_add/tool.py` 迁移为 `ImageAddInputs/ImageAddOutputs`（upstream 非 0 原样转发、inplace 保持 ndarray 对象同一性、错误显式 Missing）；`counter.py` 保持类入口，execute 用 `CounterOutputs`，initialize/release 用 `LifecycleStatus`（宿主要求生命周期返回恰好两个状态键）。
- 已完成：`dist/visiontools-deploy`（rtx40-build）与 zip 重新打包（22 工具，含 SDK 子包）；VisionRT `build/tools-package` 精简夹具由新版部署包重建（21 工具、preprocess+metrology、剔除 yolo）。
- 未完成：无（T36-19 联合发布在 VisionRT 侧推进）。

**验证证据**
- `ctest -R "test_preprocess_contract|test_upstream_status_contract|test_python_tools"` → 3/3 Passed；`python tests/test_python_tools.py` → 14/14 OK。
- 安装目录导入冒烟：dist 内 `visiontools_sdk` 直接导入，InputReader/OutputWriter/状态键守卫行为正确，不依赖开发机 PYTHONPATH。
- 宿主复验（VisionRT）：`python_contract`（完整 dist 包）与 `installed_python_author_contract`、`installed_pipeline_contract`、`cpu_tools_contract` 等 7/7 Passed；`python_shutdown_contract`、`runtime_python_lazy_contract` 等 6/6 Passed。

**下一步**
- 无新增动作；GPU 真实权重端到端联调仍受外部权重阻塞（见顶部待裁决）。

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


