<!-- 维护原文报告、方法复用与本项目增量的边界，编号不等同实验编号。 -->
# 现有文献：按解决的计算问题组织

核对日期：2026-10-20。以下是本项目最直接的方法起点与反证来源，不是整个视觉可解释性
领域的完整综述。论文发现属于作者报告；本项目复现状态单独链接实际run。链接指向原始论文/作者代码。

## 一、从视觉绑定到空间地址

<a id="r1"></a>
### R1 · Visual Symbolic Mechanisms: Emergent Symbol Processing in Vision Language Models

[论文](https://arxiv.org/html/2506.15871) · [方法 §§3–4](https://arxiv.org/html/2506.15871#S3)

作者从场景描述中的绑定问题出发，用 PCA/RSA、head-level 因果中介分析与原型 ID 干预，
建立 ID 获取、选择、特征读取的解释。§4.4 已把描述任务中的 ID 用于“直接方向＋参照”的颜色任务。
该迁移把正确目标位置的原型注入选择头；它验证下游可复用接口，没有恢复关系怎样算出应注入的目标。

**复用**：原任务、原型提取、clean/donor 对和读取接口。
**推进点**：恢复关系条件接入地址选择的原生计算，而非重复 ID 存在与跨任务复用。
§4.3 的视觉 key 干预在 RoPE 前；本项目搜索线的 post-mRoPE cache 替换是另一端口，须分别报告。

<a id="r2"></a>
### R2 · Linear Mechanisms for Spatiotemporal Reasoning in Vision Language Models

[论文](https://arxiv.org/html/2601.12626) · [作者代码](https://github.com/Raphoo/linear-mech-vlms)

作者用位置/属性交换追踪信息流，对对象词激活按位置中心化、跨对象平均提取 Spatial ID，
再以定向替换检验空间信念的变化。§2.3 的线性 OV 说明是带注意力集中等假设的 theoretical sketch，
不是已恢复的参照—关系组合算法。

**复用**：地址基候选与内容变化对照。
**推进点**：对象词携带空间成分，不等于已解释“给定参照与关系，怎样生成目标地址”。
提取出的 ID 不自动是纯位置变量。

<a id="r3"></a>
### R3 · The Dual Mechanisms of Spatial Variable Binding in Vision–Language Models

[论文](https://arxiv.org/html/2603.22278)

作者使用三个对象的横/竖布局，结合位置 probe、视觉条带互换及排序成分削弱，研究视觉
排序信息的来源与语言主干的补充作用，并扩展到 What’sUp 合成图及 COCO-spatial。
当前 v3 §5.1 已用末 token 干预区分目标位置/属性迁移，附录 C.9 已有二维网格 probe。
因此网格、可解码地址、位置/属性区分本身都是前作基础，不作本项目核心创新。

**复用**：Squares/Shapes/Objects任务与来源干预；[作者源码](https://github.com/Nix07/spatial-variable-binding)。
本项目已完成§5.1 Squares主复现与适配对照，见[E02/E03结果](../reports/e02_e03_2026-10-20.md)。
源码实际操作追加生成标记后的最后位置，图示is不能直接当作物理token索引。
**推进点**：区分视觉端已有结构和关系触发的在线运算。条带可解码不等于存有且使用了
方向邻接表；排序削弱后的剩余信息也要结合其具体干预定义解释。

## 二、空间角色、读取路径与反证

<a id="r4"></a>
### R4 · Who Is Left of Whom? Tracing Spatial Evidence and Role Binding in Relative-Position Reasoning

[论文](https://arxiv.org/html/2609.35486)

作者用对象位置交换、target/reference 角色翻转、源端到查询端 patching，以及 location-ID
和角色方向干预，连接空间证据、查询对象位置与关系预测，并检查自然图像上的成对一致性。

**复用**：角色对照、源端到查询端传播检验。
**推进点**：关系判断与“由关系寻找另一对象再读取属性”不同；后者需要解释目标选择链。
位置/角色方向本身也不能代替具体算子的恢复。

<a id="r5"></a>
### R5 · Retrieval Heads Meet Vision: Uncovering How VLMs Locate and Extract Visual Information

[论文](https://arxiv.org/html/2608.27417)

作者在 grounding 任务中比较 head 评分，使用输出位置到真值区域的 attention mass，再用
头消融检查读取的因果作用及迁移。

**复用**：候选读取头的定位与同数量随机头对照。
**推进点**：重要读取头可能消费上游已形成的地址；定位头不能说明地址在该头中生成。
本项目不默认原文每种模型的头编号可迁移到 Qwen2-VL。

<a id="r6"></a>
### R6 · Matching Object or Relation? Tracing Abstract Reasoning Inside VLMs

[论文](https://arxiv.org/html/2610.07646)

作者构造对象表面匹配与关系匹配的竞争，结合 RSA 和逐头 activation patching 分析两类
信号及其层级分布。

**复用**：竞争解释、因果定位和模板/输出路径审查。
**推进点**：关系相关组件的存在不是其组合方式的完整解释；本项目检验关系如何接入地址。

<a id="r7"></a>
### R7 · Decodable Is Not Grounded: A Vision-Ablation Arbiter for VLM Spatial Reasoning

[论文](https://arxiv.org/html/2606.31257)

作者用真实图与视觉消融比较，检查空间信息的可解码性、可操控性和行为中的视觉依赖是否一致。

**复用**：空白/错配图像对照和 held-out 超参数选择。
**含义**：本项目要检验干预是否改变了图像依赖的计算，不能把 probe 或 steering 成功直接
称为原生视觉机制。其特定模型/轴向的发现不外推为本项目的实测事实。

<a id="r8"></a>
### R8 · LLMs Process Lists With General Filter Heads

[论文 v3](https://arxiv.org/html/2510.26784v3)

作者在列表任务中分析筛选条件和可复用的 filter heads：跨候选集合迁移谓词 query、交换候选 key，
以及在问题前置时检查预先形成的候选匹配信息。输入顺序影响条件在何处进入计算。

**复用**：条件信号与候选交互的路径分析，以及问题前后顺序控制。
**边界**：发现视觉任务也有过滤头增量较弱。这里的关键是当前图像决定的参照绑定，
怎样与关系共同产生筛选条件；仅显示新 query 在旧候选上可用尚未回答该问题。

<a id="r9"></a>
### R9 · Decomposing Query-Key Feature Interactions Using Contrastive Covariances

[论文](https://arxiv.org/html/2602.04752)

作者比较匹配/不匹配的 QK 对，用对比协方差及分解识别特征交互子空间。

**复用**：在受控数据上定位参照/关系相关的 QK 方向。
**边界**：低秩交互是分析结果；需要原生路径与干预预测，才支持某个关系运算解释。

<a id="r14"></a>
### R14 · Pathways of Visual Information Flow in Vision-Language Models

[论文](https://arxiv.org/html/2607.03358v1)

作者比较图像、查询文本与末位置移植，区分直接视觉读取与文本中介，观察到路径随任务、
数据和提示改变，部分损伤会诱发备用路径。原文查询集合不包含末位置及所有特殊／聊天token，
与本项目包含 vision_end 和 assistant 标记的完整后缀不是同一集合。

**复用**：自然来源与干预后重路由分开测量。后缀优于末位置这一宽泛现象已有方法积累。
**推进点**：解释动态参照条件的生产计算及其实际消费路径；视觉背景广泛携带信息也不直接等于文本后缀依赖。

## 三、从局部归因推进到联合集合搜索

<a id="r10"></a>
### R10 · AtP*: An Efficient and Scalable Method for Localizing LLM Behaviour to Components

[论文](https://arxiv.org/html/2403.00745)

作者针对梯度归因的 false negatives，处理 attention saturation 和直接/间接效应抵消。
Q/K 修正显式重算首个 attention softmax，再近似后续计算。

**复用**：强单项评分基线、精确首个非线性＋下游局部近似的接续。
**推进点**：共同替换多个槽位时的集合归一化效应及反馈搜索，不把已有 Q/K 修正重新命名为贡献。
“迭代 AtP*”在本项目中是适配基线，不冒称原文已有同一集合搜索算法。

<a id="r11"></a>
### R11 · Finding Most Influential Sets

[论文](https://arxiv.org/html/2606.05919)

作者将一类 leave-set-out 影响写为单个线性分式，用 Dinkelbach 参数化将子问题化为 top-k。

**复用**：单分式精确原语和小实例核对。
**边界**：原任务是统计影响分析；共同集合的多个分式不能用一次 top-k 解决。有限终止条件
不等于一般共同搜索有多项式最坏复杂度，也不是注意力搜索的现成性能结果。

<a id="r12"></a>
### R12 · Billionnet–Djebali (2006), Résolution d’un problème combinatoire fractionnaire par la programmation linéaire mixte

[官方摘要与元数据](https://www.numdam.org/articles/10.1051/ro%3A2006013/)
· [原文 PDF](https://www.numdam.org/item/10.1051/ro%3A2006013.pdf)
· DOI: 10.1051/ro:2006013

官方摘要说明：二元分式和已有混合整数线性表述、通用求解和专用 branch-and-bound 路线。
本次核对到官方摘要/元数据，尚未逐式复核其完整 MIP 实现。

**复用**：作为通用求解路线，而不是宣称首次研究分式和。
**后续责任**：求解器实现前精读原文、选定可复现的通用基线；当前的专用界推导独立写在 05。

<a id="r13"></a>
### R13 · Finding Transformer Circuits with Edge Pruning

[论文](https://arxiv.org/html/2406.16778)

作者联合优化连续边掩码，以稀疏电路的整网行为忠实度为目标。

**复用**：在相同 K 替换端口上实现连续掩码优化的强适配基线。
**边界**：原文边集合与本项目的缓存槽位不是同一干预单位；适配后须统一 donor、预算和评价，
不能直接比较原文稀疏率或声称我们的解析式覆盖任意残差边。

## 四、实现来源与数据入口

<a id="c1"></a>
### C1 · Qwen2-VL-7B-Instruct

[模型说明](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
· [配置](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/blob/main/config.json)
· [Transformers v4.57.1 参考实现](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py)

配置入口显示语言主干有 28 层、28 个 query 头、4 个 KV 头；每个 KV 头对应 7 个 query 头。
视觉输入经过处理与空间合并，不能把原始像素 patch 序号直接当成语言缓存槽位。
本次配置查阅入口为 main，正式运行必须固定 checkpoint/processor revision。
当前实验确实使用 Torch2.8.0／Transformers4.57.1；GPU继承环境和processor来源按run记录，
不能只凭main配置链接代替固定revision。新端口仍需独立验收。

What’sUp/COCO-spatial 的首轮外部扩展沿用 R3 的数据构造与协议，并在实际导入时核对
数据版本、可用标注和许可证。COCO-spatial 的关系分类不能代替目标寻址的验证。

## 本轮覆盖与下一步

上述文献足以建立首批学习性复现和判别实验。它没有完成所有近期空间推理与组合优化方法
的穷尽筛查，因此当前不作“首次发现某机制”或“新分式算法”的发表级新颖性结论。
开始某一基线实现前，核对对应代码/版本与实际接口；新证据改变问题时更新相关章节，
而不是增加条件来保住预定假说。
