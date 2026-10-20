<!-- 维护唯一实验编号、依赖树和阶段状态；操作细则由对应协议维护。 -->
# 实验设计：闭合动态参照条件到属性读取的原生计算

更新：2026-10-20。本章是实验编号和依赖关系的唯一登记表。详细操作在 `protocols/`，
原生端口在 [03](03_interfaces_and_notation.md)，计算假说在 [04](04_mechanism_hypotheses.md)，
证据与决策在 [07](07_outcomes_and_decisions.md)，首批清单和预算在 [08](08_resources_and_reproducibility.md)。
本次整合只更新文档与代码组织；以下新增科学节点均未执行。

## 编号：一个实验一个名字，协议版本与执行另记

统一标识为 `编号_English_Name`：`B` 表示行为与复现，`M` 表示机制辨识，`V` 表示外部验证，
`S` 表示独立搜索。编号在各类内稳定；不再使用 E03a/b/c/d、N1 或聊天序号编排新实验。
Python 子包使用同名小写，如 `m06_condition_composition`；只有真实实现后才建立子包。

- `experiment.id`：新执行使用完整标识；当前入口不接受旧 ID；历史 run 的配置副本保持原样。
- `protocol_version`：定义数据、语义与操作版本；新版字符串按配置表使用；命名迁移不改变数据与操作语义，旧版本只保存在证据记录中。
- `run ID`：一次真实执行。相同实验可有多个版本与 run；未执行计划没有 run ID。
- `case`：同一科学问题内部的条件分支，例如 `scene_conditioning`，不再另叠一层实验编号。
- P1/P2/P3 仅表示阶段，不是实验 ID。状态分别记录“已实现”“已执行范围”“科学结论”。

## 实验登记表

| 标识与协议 | 阶段 | 2026-10-20 状态 | 主依赖／优先级 |
| --- | --- | --- | --- |
| [B01_Behavioral_Baseline](protocols/B01_Behavioral_Baseline.md) | P1 行为 | 已实现；discovery 8族已执行；原 validation/test 未执行 | 历史行为证据，按具体需要复用 |
| [B02_Binding_Replication](protocols/B02_Binding_Replication.md) | P1 复现 | 已实现；主复现、措辞适配及冻结比较已执行 | 历史因果锚点，无需重跑全扫描 |
| [B03_Reference_Disambiguation](protocols/B03_Reference_Disambiguation.md) | P1 行为鉴别 | 计划；未实现、未执行 | 立即；排除方向选固定边界 |
| [M01_Interface_Diagnostics](protocols/M01_Interface_Diagnostics.md) | P2 接口 | 线性／二维整状态诊断已实现并执行；中继 case 未执行 | 历史接口约束；中继为低优先级辅助 |
| [M02_Path_Diagnostics](protocols/M02_Path_Diagnostics.md) | P2 路径诊断 | 后缀组恢复／条件旧写入已实现并执行 | 历史效应依赖，不等于自然电路 |
| [M03_Native_Instrumentation](protocols/M03_Native_Instrumentation.md) | P2 测量 | 计划；所需新端口未实现、未验收 | 立即；离线实现可与 B03 并行，GPU验收遵循08门槛 |
| [M04_Attribute_Readout](protocols/M04_Attribute_Readout.md) | P2 原生路径 | 计划；未实现、未执行 | B03 可解释域＋M03；核心 |
| [M05_Selection_Carrier](protocols/M05_Selection_Carrier.md) | P2 原生路径 | 计划；未实现、未执行 | M04；核心 |
| [M06_Condition_Composition](protocols/M06_Condition_Composition.md) | P2 组合计算 | 计划；未实现、未执行 | M05；核心科学节点 |
| [M07_Compact_Explanation](protocols/M07_Compact_Explanation.md) | P2 紧凑解释 | 计划；未实现、未执行 | M06 已定位的真实依赖 |
| [M08_Local_Replacement](protocols/M08_Local_Replacement.md) | P2 原位检验 | 计划；未实现、未执行 | M07 的中间响应预测 |
| [M09_Frozen_Prediction](protocols/M09_Frozen_Prediction.md) | P2 冻结验收 | 计划；未实现、未执行 | M08；新场景族及未拟合干预 |
| [V01_External_Validation](protocols/V01_External_Validation.md) | P3 外部范围 | 计划；未实现、未执行 | M09 形成解释后 |
| [S01_Joint_K_Search](protocols/S01_Joint_K_Search.md) | 独立方法线 | 计划；未实现、无性能结果 | 独立授权、独立验收 |

### 历史映射：保留旧证据身份

| 旧名称／版本 | 当前归属 |
| --- | --- |
| E01、E01.1/E01.2、`e01-v1` | B01_Behavioral_Baseline；两类指标共享逐问输出 |
| E01_smoke、`engineering-v1` | B01 包内 engineering smoke；新版入口仍是工程检查，不并入正式证据 |
| E02、`e02-v1`、`e02-adjacent-v1` | B02_Binding_Replication |
| E03a、`e03a-v1/v2`、`e03a-grid-v1` | M01_Interface_Diagnostics |
| E03b、`e03b-mediation-v1`、`e03b-conditional-freeze-v1` | M02_Path_Diagnostics |
| 旧 E03a 的新端口计划 | M03_Native_Instrumentation |
| 旧 E03b 后续原生路径计划 | M04／M05／M06；旧诊断不能充当其完成证据 |
| 旧 E03c | M07_Compact_Explanation＋M08_Local_Replacement |
| 旧 E03d | M09_Frozen_Prediction |
| 旧 E05 | V01_External_Validation |
| 旧 E04、E04a–c | S01_Joint_K_Search，内部 case 为 port／solver／benefit |

当前配置、协议和代码入口统一使用新版，不保留 E 编号兼容层。run目录、报告标题和来源快照中的 E 编号不改写。
上表是证据追溯映射，不是可执行别名。原报告中的“下一步”是当时建议，当前决策以本章和07为准。

## 依赖树与可并行部分

```text
B03 反边界行为 ──可解释行为域──┐
                              ├─ M04 属性读取 → M05 条件载体
M03 原生端口验收 ─────────────┘                    │
                                                   ▼
                                  M06 参照与关系的实际组合
                                    ├─ scene_conditioning（条件触发）
                                    ├─ precomputed_neighbors（条件触发）
                                    └─ parallel_and_backup_routes（条件触发）
                                                   │
                                                   ▼
                                  M07 紧凑解释 → M08 原位检验
                                                   │
                                                   ▼
                                        M09 冻结预测 → V01
```

B03 若显示稳定边界启发式，可沿同一树解释该实际启发式，但须收缩主张；不得仅筛答对子集声称一般邻接运算。
M01 的 `address_relay` 只在原生路径提示可迁移选择状态且该主张需要它时接入，不是 M06/M07 前置条件。
S01 不进入机制依赖链，也不能补偿 M06–M09 缺失。

## 闭环的完成单位

需要闭合的是：**实际图像依赖的参照绑定＋关系输入 → 条件组合 → 完整候选竞争 → 属性写入**。
允许多位置、多头、多个层协作；独立目标地址、纯空间子空间、单头和仿射形式均不是前提。
找到重要头、可迁移 query 或后缀恢复效应只是定位成果。

每次向上游扩展必须指明：当前最早预测失败的接口、缺少的具体输入，以及增加该输入后不同的预测。
不以找全所有头、最早可解码位置或逆向整个视觉编码器为目标。预算内两轮边界扩展仍无可区分预测时，
做不可辨识审查，而非自动扩大状态。

## 共用数据与证据规则

1. 同图问题、配色、镜像和位置变体按 scene-family 分组，不跨 discovery/validation/test。
2. 同参照改关系、同关系改参照、不同条件同目标、目标／非目标属性变化形成主要自然对照。
   问题不变而图中参照移动用于检验图像依赖绑定；换色不自动是内部 V-only 操作。
3. B03 六个合法问句组合全部进入发现域。M09 新族是新实例泛化，预注册干预是因果响应预测；
   只有从未参与路线选择的结构条件才能称组合外推。新布局若用于辨识，就归发现域。
4. 保存全量自然错误、原始 greedy 与候选概率；正确基线子集另列覆盖。位置迁移按 `outcome`／
   `greedy_rate` 计数，不用 `candidate_prediction` 替代实际输出。
5. 图中对象框是来源定位线索，不保证对应 token 只包含该对象；背景／文本／特殊 token 仍保留。
6. 四对象排除固定边界，不自动区分像素移位、序号运算和邻接表；2×2 的行／列复制捷径仍是有效竞争解释。
7. 历史输出可复用；所需新激活未持久保存时需重新捕获，同一捕获同时作为自然基线，不重复计算。

## 四类科学出口与共同指标

各协议使用 `support`、`refute`、`uncertain`、`equivalent` 四类出口，避免把不兼容混合状态判为机制不存在。
支持需要自然依赖、局部来源干预与预先预测的中间响应共同成立。反驳针对某个具体形式；
等价合并要求覆盖已声明的完整接口、依赖和干预响应，不只比较候选内 softmax。

记录实际输入、任务引起的组件写入变化、完整可见来源的分数差／总权重、pre-O 与属性写入，最后才是答案。
近零效应报告绝对误差；大公共残差下的高余弦相似度不足以验收。容差在 validation 冻结：高于端口误差、
小于能区分最强竞争解释的差异；若无可用间隔，结论是当前不可辨识，不事后放宽。

自然错误参与预测；没有自然错误时不声称已解释错误机制。保留未解释残差可维持接口兼容，
但其中承担的条件计算不能算入已解释贡献。测试后修改模型则升协议版本，旧 test 转为诊断证据。

## 实施入口

先完成 [08 的第一批清单](08_resources_and_reproducibility.md)：B03、M03 和通过门槛后的 M04 有限自然捕获。
首批不含头／边全扫描、地址中继、算子拟合、S01 或新模型。离线工程检查不产生模型机制结果；
GPU 执行、模型下载、租用资源与远程发布分别需要授权。
