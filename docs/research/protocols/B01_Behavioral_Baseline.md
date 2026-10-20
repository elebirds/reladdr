<!-- 固定 E01 v1 的任务、数据、留出、评价和执行边界；作为该入口的唯一详细协议。 -->
# B01_Behavioral_Baseline — 分组行为基线

2026-10-20。阶段 P1；旧证据为 `E01/e01-v1`；当前ID为本页标题、协议 `b01-v1`。已实现，历史discovery已执行；原validation/test未执行。
能力分解与成对依赖共享逐问输出，不重复推理。历史 `E01_smoke/engineering-v1` 只作工程证据；当前smoke使用 `b01-engineering-v1`。
以下保留该历史协议的数据和执行契约；新四对象鉴别见 [B03](B03_Reference_Disambiguation.md)。

## 问题、文献与模型

能力分解测单对象感知、位置读取和关系条件读取；成对依赖测输出是否跟随图像、参照与关系。
R3 已在三对象任务和当前模型上报告高准确率，R1 §4.4 已有方向关系颜色读取。这里是能力
复现与边界检查；布局、模板和像素规则作了适配，不冒称逐像素复刻论文，不以网格等主张创新。

模型为本地固定 revision 的 Qwen2-VL-7B-Instruct，BF16、冻结参数、SDPA、batch=1，
无量化/CPU offload。现有 processor 预算约 448×448，实际 token/grid 逐问记录。
greedy 最多 8 个新 token。所有条件统一系统提示 `You are a helpful assistant. You respond in one token.`，
避免某模板缺少回答格式约束。工程 smoke 不加此提示；不下载权重，不自动更换模型。

## 场景与问题

布局为横排 1×3、竖排 3×1、2×2、3×3、4×4。完整图 448×448，白底；对象 bbox 边长
56，密度增加时不缩小。标准中心按 28 像素设定，每族整体平移最多 5 像素，保留规则邻接，
并避免族间完整反事实图像复用。每图颜色+形状组合唯一，颜色允许重复；无空位、无边界环绕。

| kind | 实际输入和提示 | 含义 |
| --- | --- | --- |
| recognition | oracle bbox 周围 112×112 单对象白色补边裁剪，问颜色 | 已给位置的识别上限，不是整图参照绑定 |
| absolute | 完整图；1D 极值/中间或 2D 明确行列，问颜色 | 位置条件属性读取 |
| relation | 完整图；唯一颜色+形状参照及直接相邻方向 | 参照绑定、关系选择与读取的联合表现 |

裁剪避免重复形状时把答案颜色写入识别提示；裁剪答对不证明整图识别无误。
模板分别报告：`relation_first` 为 “The color of the shape immediately to the left of the green circle is”；
`reference_first` 为 “Look at the green circle. What color is the object immediately to its left? Answer with one color word.”。
其他方向替换介词。自然语言顺序不自动证明内部算法。

## 固定场景族与组合留出

首批预生成 discovery=8、validation=4、test=4 个独立族，**只运行 discovery**。
同族布局、多问、裁剪和视觉变体不跨 split。版本 `relational-grid-v1`，seed=19。
有限三对象属性锚点池预先固定 split；扩量不重分已有族。当前最多支持 72/24/24 个
对应 split 的族，超出时需修订生成协议，不悄悄复用锚点。

组合 seed=20261020。1D/2×2 是 shared 锚点；3×3/4×4 每方向合法参照格中约 1/6 为
validation_only、1/6 为 test_only，其余 discovery，同一规则跨族/模板固定。
Discovery 只问 shared/discovery；validation 问 shared/discovery/validation_only；
test 问 shared/discovery/test_only。直接位置/识别控制不受关系组合留出限制。
本版只声明场景族与参照格—关系组合留出；形状、颜色和五种布局都可能跨 split 出现，
不据此声称全新类别或未见布局泛化。完整真值可保存，未请求 split 不调用模型。
首批是 discovery pilot，不是精度保证；
调整后修订协议版本，test 保留到冻结检验。

## 固定问题下的视觉对照

- `original`：原图。
- `blank`：同尺寸全白，没有目标真值，不报告准确率。
- `target_recolor`：参照和位置不变，改变目标颜色；优先用同形未占用色。六色占满时与
  同形非参照对象交换颜色，记录两个变化对象，不声称总是只改一个对象。
- `position_swap`：交换目标和另一非参照、不同颜色对象的位置，参照描述/位置保持有效。
  是有真值的布局反事实，不称为独立无关照片。

每个条件保存实际输入、对象修改及重新计算的 target/答案；位置交换不沿用旧 target ID。
重复颜色仅记录答案兼容对象集合，不推定唯一误选对象。

## 评价和统计

1. 保存六色原始输出、单词格式与词表成员、大小写别名合并后的完整词表下一 token 概率、
   候选 argmax、token 数和时间。分词边界上非单 token 明确失败，不截断。
2. 原答案/foil 在问题生成时固定，视觉条件间不改，不每次取当前最强错误答案。原图与
   反事实按实际新答案算 greedy/候选准确率，空白仅看原标签匹配和固定 margin。
3. 切片按 split/布局/kind/模板，方向和组合 partition 另列。视觉配对按同一问题；
   参照变化按同图同关系中参照槽位顺序比较相邻项；关系变化比较同参照合法的左右/上下。
   不按模型对错挑配对，报告两边均正确、预期/实际颜色改变和固定 margin 变化。
4. 原图记录 reference crop、target crop、absolute、relation 的正确性四元组，只作行为诊断。
5. scene-family cluster bootstrap 1000 次，统计量为相应总成功数除总数；报告族数与问题数。
   少于 2 族不给区间；不能把数千同族问答当数千独立样本。
6. 保存真实调用、token、耗时、阶段时间与 GPU 峰值。首批 tracking/OSS 关闭。

## 执行与产物

先在 CPU 准备和检查真实渲染：

```sh
uv run --locked python -m reladdr.experiments.b01_behavioral_baseline configs/B01_Behavioral_Baseline.yaml --prepare-only
```

复制为忽略的 `configs/B01_Behavioral_Baseline.local.yaml`，填本地 Instruct 路径和实际 revision。
可通过 `model_source_manifest` 指定既有逐文件来源 JSON，运行时复制到 provenance；
上游下载标签 master 不冒称固定快照，权重与 processor 版本按清单分别解释。完整环境：

```sh
uv sync --locked --extra gpu
uv run --locked --extra gpu python -m reladdr.experiments.b01_behavioral_baseline configs/B01_Behavioral_Baseline.local.yaml
```

已有 AutoDL 镜像沿用 [GPU 初调](../../gpu_smoke.md) 的 `--no-sync` 环境，明确记录实际
Python/分发版本，不称干净全锁环境。入口不启动新机器，也不上传 OSS。

产物包括 `dataset/scenes.jsonl`、`questions.jsonl`、`manifest.json`、`preview.png`、原图与实际
输入，以及 `baseline/predictions.jsonl`、`pairs.jsonl`、`summary.json`、`report.md`。
逐问即时刷新；故障保留已写结果与 failed 状态，不补造 summary/自动续跑。
prepare-only 记录 evaluation_executed=false，不记模型结果；源码修改及未跟踪源码由运行层保存。

## 怎样改变行动

识别上限差先查渲染/颜色/预处理；位置读取差定位提示难度或空间绑定；前两者稳定而关系差，
再决定相应原生机制节点的行为域。保留低表现方向，不将最强方向改名为整体。数据/行为边界和不确定性
足够支持下一项判断即可推进，不设统一准确率门槛。

首批执行结果见 [2026-10-20 discovery 记录](../../reports/e01_discovery_2026-10-20.md)。

当前只使用 `B01_Behavioral_Baseline/b01-v1`；旧入口已移除，历史run不改写。
工作量按实际问题数及视觉条件求和；本协议没有新增执行预算，旧 discovery 不重跑来制造新证据。
