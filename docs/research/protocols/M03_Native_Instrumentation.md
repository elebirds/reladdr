<!-- 固定原生 Q/K/V、mRoPE、GQA 和完整归一化端口的工程验收；不把端口正确当作机制发现。 -->
# M03_Native_Instrumentation — 建立可回溯的原生测量端口

协议版本：`m03-native-ports-v1`，2026-10-20。**计划；新增能力尚未实现或执行。**
现有无缓存残差/组件移植已经执行，不等于本页的逐头 Q/K/V 和分数重建已经通过。
本项只验收测量与干预实现，不筛选重要头、不拟合关系算子。
总预算见 [08 首批清单](../08_resources_and_reproducibility.md)；
科学接口见 [03](../03_interfaces_and_notation.md)。

## 1. 已有端口与实际缺口

以本轮审读的 `models/qwen2_native.py`、`interventions/hooks.py` 为实现基线：

| 当前 `Site.component` | 原生位置 | 已有能力 | 本协议新增验收 |
| --- | --- | --- | --- |
| `residual` | decoder block 输出 | 指定 token 行整状态 capture/replace | 复用既有证据，不重扫 28 层 |
| `attention` | `self_attn` 输出 | 多头合并后的输出 | 不把它命名为单头写入 |
| `mlp` | `mlp` 输出 | 指定 token 行输出 | 为后续读取端记录保留 |
| `heads` | `o_proj` 输入 | 可按 query head 选择列 | 核对头切分与 `W_O` 写入 |
| `query` | `q_proj` 输出 | 整行，mRoPE 前 | 新增逐 query-head 切片；记录真实旋转后 Q |
| `key` | `k_proj` 输出 | 整行，mRoPE 前、GQA 展开前 | 新增逐 KV-head 切片；记录真实旋转后 K |
| `value` | `v_proj` 输出 | 整行，GQA 展开前 | 新增逐 KV-head 切片；V 不施加 mRoPE |

现有 `Site.head` 只允许 `heads`，不能把 `query/key/value` 的整行捕获描述成已经具备逐头
干预。现有原生前向使用 `use_cache=False`，并从真实最终 norm 输出计算末位置 lm_head；
这条重算路径保留。当前 `forward_calls` 只在成功返回后增加，不能用于失败调用预算。

后续实现应在同一个模型 adapter 内补齐下列能力，不能把模型特有逻辑复制到实验脚本：

1. 对 Q 与 K/V 分别检查合法头数、列范围、dtype、device 和实际序列位置。
2. 从 Transformers 4.57.1 的实际前向取得旋转后 Q/K、position IDs、cos/sin、
   `mrope_section` 和有效 mask。仅离线再算一个“类似 RoPE”的版本不够。
3. 在不替换 SDPA 为另一后端的情况下，重建选定 query 行的全候选分数、归一化权重和
   来源贡献，并与真正进入 `o_proj` 的头输出比对。
4. 记录每次 attempted/succeeded/failed forward；异常前分配或调用模型也计为尝试。
   每次尝试即时写记录，恢复 hook 后才允许下一条输入。

逐头 projection 替换是本批新增的干预端口；旋转后 Q/K 首批只要求可靠捕获。后续若需要
新增 post-mRoPE patch，应独立声明、验收其物理位置，不能复用 pre-mRoPE 的名称与证据。
S01 的 post-mRoPE、pre-GQA **K-cache** 集合接口仍未实现，也不由本页宣布完成。

## 2. 完整注意力计算的验收对象

使用模型配置的实际值。本模型预期 28 个 query heads、4 个 KV heads、每头 128 维；
运行时必须记录并断言，不能在通用切片函数中写死。每个 KV 头对应 7 个连续 query heads：

\[
g(h)=\left\lfloor h/7\right\rfloor,\qquad
s_{tj}^{h}=\frac{q_{t,\mathrm{post}}^{h\top}k_{j,\mathrm{post}}^{g(h)}}{\sqrt{d_h}}
            +m_{tj}.
\]

`post` 指本次原生 mRoPE 后，\(m\) 是本次真实 causal/padding mask。GQA 不把 4 个 K/V
头误当作 28 个独立源端口。改一个 native KV head 时，其影响覆盖共享它的全部 7 个
query heads；不能只报告末位置或其中一个头。
若 SDPA 以 `attention_mask=None` 配合 `is_causal` 执行，记录该标志并恢复其有效可见范围，
不能把空 mask 解释为全序列双向可见。

\[
\alpha_{tj}^{h}=\operatorname{softmax}_{j\in\mathcal K_t}(s_{tj}^{h}),\qquad
y_t^h=\sum_{j\in\mathcal K_t}\alpha_{tj}^{h}v_j^{g(h)}.
\]

\(\mathcal K_t\) 是该行可见的**全部原生 key**，包括系统文本、图像、普通问题文本及特殊
标记，不是只对视觉 token 做 softmax。视觉贡献定义为
\(y_{t,\mathcal V}^h=\sum_{j\in\mathcal V}\alpha_{tj}^{h}v_j^{g(h)}\)，继续使用同一个完整分母。
source mass、候选分差和背景贡献都由这份原生权重计算。

保存互斥、穷尽的来源分组，至少区分实际视觉 token、普通文本、特殊/chat token。
图中对象框与视觉 token 的覆盖是另一个映射：部分覆盖 token 和背景须单列，不能因
“看起来不含对象”而从分母或 V 中删除。对象级归因所需的细分留给 M04，不增加本批输入。

头写入为 \(w_t^h=W_O[:,I_h]y_t^h\)。所有头写入之和应恢复 attention 投影输出；
若实际 `o_proj` 有 bias，只加一次并单列。它不包含 attention 残差加法或随后的 MLP。
重建需说明每一步使用的 dtype；FP32 重建只是数值参照，不改变 BF16/SDPA 原生执行。

## 3. 离线实现先于 GPU 端口校验

本节离线实现可与 B03 生成器开发并行；第 4 节真实 GPU 验收按 B03 覆盖门槛推进。

CPU 测试覆盖形状、空/越界位置、重复位置、Q 与 KV 头边界、无广播替换、donor 不变、
其他列不变、临时 hook 成功/异常清理，以及 GQA 共享映射。小张量测试覆盖：

- mRoPE 前后标签和旋转次数；实际 adapter 不能对已旋转 K 再施加一次旋转；
- 包含视觉、文本和 mask 的完整 softmax，与来源分组求和；
- 多头写入与完整 `o_proj` 输出的一致性；
- 尝试/成功/失败计数及异常恢复记录，不能只复用当前成功计数器；
- 无缓存前向重新计算所有受影响的后续 token/层，不能混用旧 suffix 状态。

这些测试不得下载或加载模型、联系远端服务。真实模型权重、token 展开、SDPA 路径与
mRoPE 捕获的验收仍需下面的有限 GPU 检查；mock 通过不替代它们。

## 4. 固定小样本和 34 次尝试上限

在 B03 通过操作性覆盖门槛、离线测试通过后，仅取两条固定输入，不按回答正确与否换样本：

- A：`b03-s20261020-f0000-c0`，`reference=1, left`；
- B：`b03-s20261020-f0001-c0`，`reference=2, right`。

校验层固定为零起点 decoder block **0、17、27**。层 17 继承旧后缀效应的测量关注，
0/27 检查首末边界；这不是先验指定机制层。

每条输入最多 17 次尝试，安排如下：

| 操作 | 每输入尝试数 | 内容 |
| --- | ---: | --- |
| 无 hook 自然基线 | 1 | 完整词表 logits、greedy、实际 token/role 清单 |
| 仅捕获 | 1 | 一次前向同时捕获三层 Q/K/V、旋转后 Q/K、mask/位置信息和原生头输出 |
| 自身替换 | 12 | 3 层 × 4 端口：`Q_pre`、`K_pre`、`V`、`y_pre_o_proj` |
| 非平凡 K/GQA 路由检查 | 1 | 第 17 层用另一输入的 K 替换一个固定视觉槽位的 native KV-head |
| 人工异常 | 1 | 在已注册 hook 内抛出预定异常；计失败尝试，不产出科学预测 |
| 移除 hook 后恢复 | 1 | 无 hook 前向与该输入原基线比对 |
| **合计** | **17** | 两输入共 **34**；没有层/头搜索 |

实际顺序先取得 A/B 的无 hook 基线和仅捕获记录（共4次），再执行各自的自身替换、交叉K检查和异常／恢复，
保证每次交叉的 donor 来自已完成的自然捕获；表格按输入归账，不要求先跑完A的17次再开始B。

仅捕获与自身替换保存全部实际文本后缀 query 行，并额外保存最后一个视觉 query 行；
K/V 则保留这些行归一化所需的完整 key 范围。A 自身替换采用 query head 0 / KV head 0，
B 采用 query head 27 / KV head 3。Q 和头输出替换上述 query 行；K/V 自身替换全 key 行。
完整头数、内部头边界由离线测试覆盖，不为了补齐 28 个头另加模型前向。

非平凡 K 检查使用两次“仅捕获”记录作为 donor：A 的第一个实际视觉槽位换入 B 对应槽位
的 KV-head 0，B 的对应槽位换入 A 的 KV-head 3。替换的是 **mRoPE 前 K**；recipient
的位置编码继续由原生路径施加。对全部受影响 suffix query 行及共享该 KV 头的 7 个
query heads，预先用固定 Q、替换 K 和固定 V 预测新分数、权重及 \(y\)，与此次原生记录
比较；未共享该 KV 头的同层头输出保持不变。下游照常完整重算。

必须检查 patch 的 K 值确实进入正确端口；不能用“logits 没变”证明替换未生效。
这条有限扰动只验收工程路径，不要求翻转答案、不解释空间语义。若两输入对应 K 完全相同，
记录该路由检查没有形成非零扰动，暂停验收；不另找容易翻转的 donor。

## 5. 数值门槛与失败处理

以下容差是本协议预先固定的**工程拒绝阈值**，不是实测误差或科学效应阈值：

1. 仅捕获、自身替换、移除 hook 后恢复与原基线：全词表 logits 最大绝对差
   `≤1e-5`，且 greedy token ID 相同。原生张量相同处另报逐元素相等比例。
2. FP32 重建与原生 BF16/SDPA 的 \(y\) 和 attention 写入：相对 Frobenius 误差
   `≤0.02`，且最大绝对误差 `≤0.01 + 0.02 × max_abs(native)`。
   相对误差分母为 `max(norm(native), 1e-8)`；零/近零向量同时报告绝对误差。
3. 非平凡 K 检查使用同一重建阈值；未共享 KV 头的同层原生输出最大绝对差 `≤1e-5`。
   全来源分组求和与未分组重建用同一 FP32 计算，最大绝对差 `≤1e-5`。

按声明的 query 行、头分别验收，不能用跨行平均掩盖失败。保存每项绝对/相对误差及其
行/头/层，不能只写 pass。后续科学解释必须比较效应与实际
重建残差；在工程阈值内不意味着足以区分弱小机制效应。若所研究的候选差异与残差同量级，
返回 U，先改善测量而不是引用“2% 以内”作支持。

除表中故意触发的异常外，任何检查失败立即停止剩余 GPU 调用。先区分端口错误、dtype/
后端差异、mask/旋转错误或日志问题，再离线修复并记录版本。不得在看见结果后放宽容差，
也不默认重跑整套 34 次。至多预留 **8 次受影响检查的重跑尝试**，仍失败或所需重跑超过
此数时结束本批，提出有依据的新计划。34 次正常尝试加 8 次修复是硬上限，不是必花额度。

异常测试确认所有临时 hooks 都已移除；恢复失败时不继续在可能污染的模型实例上解释
结果。结束该实例并保留可复现记录，实例重建/额外重跑须进入新的明确预算。

## 6. 交付与进入 M04

交付实际端口清单、token/role/position/mask/GQA 元数据、三层数值检查、成功和失败日志、
attempted/succeeded/failed 计数、逐段耗时及峰值 allocated/reserved 显存。
只保存选定行/层与重建必需的 K/V；不默认常驻全层全注意力矩阵。

全部门槛通过才将该实现交给 [M04](M04_Attribute_Readout.md)。M03 的两条输入、三层捕获
不能抵扣 M04 的全层自然记录；34 次均记为工程 checks，不重复计入自然样本量。
M03 通过说明所声明的原生端口可用，不支持“Q 执行关系变换”“K 是地址”或“这些头负责选择”。
