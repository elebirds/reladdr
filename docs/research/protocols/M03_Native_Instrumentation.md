<!-- 按实际消费者递增测量能力，复用已有写入端口，不以前置平台阻塞原生机制研究。 -->
# M03_Native_Instrumentation — 按消费者补齐原生测量端口

协议版本：`m03-native-ports-v2`，2026-10-20。新增流程**未实现、未执行**。
M03 是按需工程支持，不是必须一次完成的研究阶段。先支持 M04 的已有写入端口，
读取候选确定后再补 M05 的注意力重建；节点干预和私有输入边由具体鉴别问题触发。
预算在 [08](../08_resources_and_reproducibility.md)，公共物理契约在
[03](../03_interfaces_and_notation.md)。

## 1. 当前实现和最短依赖

| 能力 | 实际现状 | 谁需要下一步 |
| --- | --- | --- |
| block-post、attention、MLP 指定行 capture/replace | 已实现 | 复用相应 hook 和恢复证据 |
| `heads`：o_proj 输入的逐 query-head 列 capture/replace | 已实现，`Site.head` 仅支持此端口 | M04 自然捕获与局部确认 |
| Q/K/V projection 整行 capture/replace | 已实现；Q/K 在 mRoPE 前，V 不旋转 | 可按形状离线切片观测，不是逐头节点干预 |
| 原生 post-mRoPE Q/K、有效 mask、来源重建 | 未实现 | M05 选定读取器后开发 |
| 逐 Q-head／原生 KV-head 投影干预 | 未实现 | 研究节点级传播时开发 |
| receiver 私有输入边、MLP gate/up、局部梯度 | 未实现 | M06 的鉴别预测需要时开发 |
| attempted/succeeded/failed 即时账本 | 当前 `forward_calls` 只计成功返回 | 首批执行前补齐，可在批次调用层包裹 |

本轮核对的 [HF 4.57.1 源码](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py)
顺序是 projection → reshape → Q/K mRoPE → attention backend → 拼接输出 → o_proj。
原生模型仍用 BF16/SDPA、batch=1、完整 `use_cache=False` 前向；表中的因果重算范围
不表示已经实现中间层快速重放。模型特有逻辑留在 adapter，不复制到各实验。

## 2. consumer_write：先支持 M04

此 case 不开发 Q/K/V 新端口或完整来源分解。复用 pre-O/MLP 捕获和写入；离线确认
指定行／头列、donor 不变、未选列不变、形状检查以及 hook 正常／异常清理。
保存 M04 后续需要的原生 y、MLP 写入和完整末位置 logits，颜色 margin 由同一 logits 计算。

首批选固定输入 `b03-v2-s20261020-f0000-c0`、问题0；不按其回答是否正确换样本。
计划六次工程尝试：

| 操作 | 次数 | 检查 |
| --- | ---: | --- |
| 无 hook 基线 | 1 | 完整 logits、greedy、真实 token/role 索引 |
| 仅捕获 | 1 | 同时捕获 M04 所需层／后缀的 pre-O、MLP；不改变 logits |
| 原始 pre-O 自身替换 | 1 | 层17、声明后缀行，首／末 query-head 可同次检查 |
| 原始 MLP 自身替换 | 1 | 层17、相同声明行 |
| 受控 hook 异常 | 1 | 在已注册 hook 内抛出，保留失败记录并清理全部 handle |
| 无 hook 恢复 | 1 | 与首条基线比对 |

实际头数、head_dim、索引由模型配置与 token 映射读取。头边界和其他列不变用离线
张量测试补齐；不为每个候选头重复整套模型检查。
捕获-only、原始张量自写和恢复的全 logits 最大绝对差要求 `≤1e-5`，greedy token ID 相同。
每次调用后核对 hook 清理，检查注册状态不需要额外模型前向。

通过后 M04 即可做自然写入筛选和局部确认。若所需端口、精度或执行上下文改变，再检查
受影响部分；不要求先通过下面的 reader_attention 或 native_nodes。

## 3. reader_attention：只为已确定的读取器观测 Q/K/V

输入是 M04 冻结的 attention 读取器和将要检验的有向问题配对。若消费者是 MLP，
先追其实际输入，不将 MLP 输出硬套成 Q/K/V。

新增观测从实际前向取得 post-mRoPE Q/K、V、position IDs、cos/sin、mrope_section、
有效可见 mask 和真正进入 o_proj 的 y。整行捕获后按配置切片即可，首轮不需要逐头 projection 改写。
捕获不得将原模型 SDPA 换成另一后端。

本模型预期 28 个 query heads、4 个 KV heads、head_dim=128；记录实际配置并断言。
GQA 映射在此配置中为 `g(h)=floor(h/7)`，V 不施加旋转。对声明的全局读取行：

\[
s_{tj}^{h}=q_{t,post}^{h\top}k_{j,post}^{g(h)}/\sqrt{d_h}+m_{tj},\qquad
\alpha_{tj}^{h}=\operatorname{softmax}_{j\in\mathcal K_t}(s_{tj}^{h}),\qquad
y_t^h=\sum_{j\in\mathcal K_t}\alpha_{tj}^{h}v_j^{g(h)}.
\]

`K_t` 是该行的全部允许来源：视觉、普通文本和特殊/chat token。保存互斥穷尽的来源分组，
来源注意力总权重为 `Σ α_j`，来源贡献为 `Σ α_j v_j`；所有贡献之和恢复 y。
对象 bbox 仅用于附加来源定位，背景和部分覆盖 token 保留在完整分母中。
头写入为 `W_O[:, I_h] y_t^h`；若 o_proj 有 bias，只加一次，不把残差和后续 MLP 算进该写入。

**全局行 mask 必须保留。**抽出一行形成 `[1,K]` 后，不可直接沿用 `is_causal=True` 的
局部三角 mask。由其真实全局位置 materialize causal/padding（以及实际启用的其他约束）mask，
局部算子设 `is_causal=False`。原生 `attention_mask=None` 时也须解析其 is_causal 语义。
参考 [HF SDPA wrapper](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/integrations/sdpa_attention.py)。

### 有限重建验收与误差背景

对实际将被交叉的两端自然输入，逐行／头比较重建 y 与原生 y、投影写入与来源求和。
报告计算 dtype。FP32 重建的预登记工程阈值沿用：相对 Frobenius 误差 `≤0.02`，
最大绝对误差 `≤0.01+0.02*max_abs(native)`；范数分母下限 `1e-8`。
来源分组与未分组的同一 FP32 求和最大差 `≤1e-5`。不以跨行平均掩盖失败。

把重建 `y000` 写回输入0、`y111` 写回输入1，记录完整 logits、颜色 margin 和 greedy 的
下游差异。这是**重建自写**，不是原始张量的严格 no-op；二者不能互相冒充。
科学效应同时相对原生和重建基线报告。若分歧预测与重建误差同量级或效应方向随基线改变，
先改善测量，不能用“工程误差小于2%”支持机制。

首个 M05 定向比较的七次账本见 M05/08：两次新观测、两次重建自写、一次恢复、两次交叉。
捕获-only 比对可复用 M04 同版本的完整 logits。若没有适用的原始 y 自写证据，另外登记该检查；
已有证据有效时不仅因换一个候选头重跑。

## 4. native_nodes 与 receiver_edges：由区分预测触发

研究共享 KV 如何传播时，才新增逐头 projection 或指定 post-mRoPE 节点干预。必须明确
pre/post-mRoPE：pre donor 使用 recipient 的原生旋转；post donor 已有旋转，不能再转一次。
一个原生 KV 槽位影响所有共享 query heads 和全部实际读取行，不能只替换复制品后这样命名。
完整无缓存节点前向还可能改变后续视觉前缀状态；它不同于 S01 固定已完成前缀的 K-cache 编辑。

需要隔离 sender 对某 receiver 的作用时，再实现私有输入边，重算该 receiver 的 norm 与投影。
节点、中介与边的定义见03。每项新增能力按其端口做最小自身替换、非零扰动路由和恢复检查，
预算随具体问题登记；不预先为所有层／头／边建设平台。

## 5. 失败处理、计账与交付

除预定异常外，端口或记录失败先停止受影响调用；检查端口、mask/旋转、dtype/backend 或清理原因。
修复后重验受影响部分并复用未失效的证据。尝试次数不足只暂停本批，不作为科学反证，
也没有“第九次复检即机制不成立”的规则。继续需要记录修复依据、增量成本和剩余额度。

每次模型尝试前登记 attempted，成功／失败均即时落盘；故意异常计失败尝试而非科学输出。
恢复失败的实例停止使用，重建实例和额外检查单独计账。
交付实际可用端口、数值误差分布、hook 状态、有效执行条件及真实成本。
M03 的工程状态和对应消费者的科学结论分别记录。
