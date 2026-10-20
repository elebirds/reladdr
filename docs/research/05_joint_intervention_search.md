<!-- 维护 S01 的数学规格与推导；执行验收在对应协议维护。 -->
# 联合干预搜索：从精确注意力响应到真实下游反馈

本章给出拟研究算法的数学规格与条件推导，没有对应实现或模型性能结果。
当前实验名为 [S01_Joint_K_Search](protocols/S01_Joint_K_Search.md)，历史名称 E04。
端口限定为 [03](03_interfaces_and_notation.md) 中的同图多问、固定层原生视觉 K-cache、
K-only、固定且不重叠分组、固定 donor、固定问题和下一答案 token。

## 1. 固定真实目标

对每个问题 $q$，预先确定正确 token $a_q$ 和 foil token $f_q$，定义：

$$
M_q(C)=\operatorname{logit}_{a_q}(C)-\operatorname{logit}_{f_q}(C),
\qquad J(C)=\sum_q\omega_q M_q(C),\quad\omega_q\ge0,
$$

$$
\min_{|C|=k}J(C).
$$

这是造成答案优势下降的干预搜索，不是修复模型的目标。权重、foil、donor 和槽位身份
在搜索前固定。逐问效应单独报告，不把共同集合称为每个问题必需的“共享核心电路”。

首版要求候选答案在实际模板/分词边界下可用下一 token 读出；模型接入时验证。多 token
答案需要另定 teacher-forced 序列目标与接口，不能将首 token 证书直接解释为整段答案保证。

## 2. 一行注意力怎样精确依赖集合

令 $s=(q,h,t)$ 为一个受影响的读取行。原生 query 为 $q_s$，来源 key/value 为 $k_j,v_j$，
其中 key 属于 $g(h)$。包含正确 mask 的分数和权重为：

$$
z_{sj}=\frac{q_s^\top k_j}{\sqrt{d_h}}+m_{tj},\qquad w_{sj}=e^{z_{sj}},
$$

$$
Z_{s0}=\sum_j w_{sj},\qquad N_{s0}=\sum_jw_{sj}v_j,\qquad Y_{s0}=N_{s0}/Z_{s0}.
$$

masked 位置的权重为 0。组 $i$ 中的 key 替换为固定 $k'_j$；令
$\delta z_{sj}=q_s^\top(k'_j-k_j)/\sqrt{d_h}$，则：

$$
b_{si}=\sum_{j\in G_i(s)}w_{sj}(e^{\delta z_{sj}}-1),
$$

$$
a_{si}^{\mathrm{vec}}=\sum_{j\in G_i(s)}w_{sj}(e^{\delta z_{sj}}-1)v_j.
$$

$G_i(s)$ 仅取该行可见、且属于其 KV 头的组内槽位。没有影响的组系数为 0。
因为组不重叠，且 Q、V 与替换值固定，有实数算术等式：

$$
\boxed{Y_s(C)=
\frac{N_{s0}+\sum_{i\in C}a_{si}^{\mathrm{vec}}}
     {Z_{s0}+\sum_{i\in C}b_{si}}.}
$$

这已经包含集合共同改变 softmax 分母的效应。实际浮点实现需报告与原生重算的误差。
可使用对该行所有候选集合固定的共同 logit shift 稳定指数；不能对每个集合改变系数定义。
分母保持正且有限是以下推导的条件，数值溢出/下溢不等于得到一个有效证书。

### 响应方向与完整接口

定义 $d_{si}=a_{si}^{\mathrm{vec}}-Y_{s0}b_{si}$，则：

$$
\Delta Y_s(C)=\frac{\sum_{i\in C}d_{si}}
 {Z_{s0}+\sum_{i\in C}b_{si}}.
$$

单行变化位于 $d_{si}$ 的张成空间；不同头/位置的分母不同，不能据此声称完整接口低维。
保留所有受影响行构成 $Y_q(C)$，接回原生 $D_q$，而不是只取一个标量读出充当充分状态。
当前不需要训练替代下游 $\Psi$。

## 3. 单项评分为何可能漏掉集合

以下为手工构造的代数例子，不是模型观测。设：

$$
F(C)=\frac{1+\sum_{i\in C}a_i}{1+\sum_{i\in C}b_i},
$$

$$
(a_A,b_A)=(10,100),\qquad(a_B,b_B)=(a_C,b_C)=(0,5).
$$

预算 $k=2$ 时，单项 $F(A)=11/101$ 优于 $F(B)=F(C)=1/6$，条件贪心先选 A，最终
得到 $11/106\approx0.1038$；而 $F(\{B,C\})=1/11\approx0.0909$ 更小。
这证明条件贪心也可能遗漏更好的集合，不证明这种现象在实际模型中的频率或收益规模。

## 4. 每轮只近似下游，不近似首个联合注意力响应

在当前集合 $C^{(r)}$，计算原生下游梯度
$g_{q,r}=\nabla_{Y_q}M_q(C^{(r)})$：

$$
\widehat J_r(C)=J(C^{(r)})+
\sum_q\omega_q g_{q,r}^\top[Y_q(C)-Y_q(C^{(r)})].
$$

分解到各行后，定义标量分子
$N_s(C)=g_{s,r}^\top[N_{s0}+\sum_{i\in C}a_{si}^{\mathrm{vec}}]$ 和
$Z_s(C)=Z_{s0}+\sum_{i\in C}b_{si}$，得到：

$$
\widehat J_r(C)=\kappa_r+\sum_s\omega_s F_s(C),
\qquad F_s(C)=\frac{N_s(C)}{Z_s(C)},\quad\omega_s=\omega_q\ge0.
$$

$N_s$ 可有正负系数，正性要求在分母。常数 $\kappa_r$ 不影响集合排序。
这是 [AtP*](02_related_work.md#r10) 所用局部近似思路的接续；本项目拟利用多个干预的
精确集合结构。内层的最优性间隙属于 $\widehat J_r$，不属于真实 $J$。

## 5. 单分式原语：参数化选择和有效下界

在可行集合族 $\mathcal F_B$，考虑 $F=N/Z$ 且 $Z>0$。给定 $\mu$，求：

$$
\phi_B(\mu)=\min_{C\in\mathcal F_B}[N(C)-\mu Z(C)].
$$

若 $\phi_B(\mu)\ge0$，则 $\mu\le\min_{C\in\mathcal F_B}F(C)$。大小约束下，线性
子问题可由排序/选择求解；Dinkelbach 式参数更新提供单分式精确原语，参照
[R11](02_related_work.md#r11)。浮点实现的界须保守处理误差，不能把近似零残差直接当严格证明。

分支已强制选入 $I$、排除 $E$，剩余可选组为 $U$，还需 $p=k-|I|$ 个。记：

$$
N(C)=N^I+\sum_{i\in C\setminus I}a_i,\quad
Z(C)=Z^I+\sum_{i\in C\setminus I}b_i.
$$

若 $p<0$、$p>|U|$ 或其他计数约束不满足，节点不可行，直接排除。
在线性大小约束下：

$$
\phi_B(\mu)=N^I-\mu Z^I+
\operatorname{SmallSum}_p(a_i-\mu b_i:i\in U).
$$

单个可行集合的 $F(C)$ 是最小化问题的上界，不是认证的下界。参数化求解的有限终止条件
不提供一般分式和搜索的多项式最坏时间保证。

## 6. 残差下界同时产生一个可行候选

对每行取认证的 $\mu_s\le\min_{C\in\mathcal F_B}F_s(C)$，并取分母上界
$\bar Z_s\ge\max_{C\in\mathcal F_B}Z_s(C)>0$。定义：

$$
r_s(C)=N_s(C)-\mu_sZ_s(C)\ge0.
$$

由于 $r_s/Z_s\ge r_s/\bar Z_s$，共同代理的下界为：

$$
\boxed{LB_B=\sum_s\omega_s\mu_s+
\min_{C\in\mathcal F_B}\sum_s\frac{\omega_s r_s(C)}{\bar Z_s}.}
$$

大小约束下，令：

$$
c_i=\sum_s\frac{\omega_s(a_{si}-\mu_sb_{si})}{\bar Z_s},
$$

$$
LB_B=\sum_s\omega_s\mu_s+
\sum_s\frac{\omega_s(N_s^I-\mu_sZ_s^I)}{\bar Z_s}
+\operatorname{SmallSum}_p(c_i:i\in U).
$$

最小 $p$ 项同时给出完整可行集合。计算它的**代理值**可更新内层上界；用上下界剪枝。
父节点的 $\mu_s$ 对子节点仍有效，可以先继承，再按收益刷新部分单项界。

在简单大小约束下，$Z_s^I+$ 最大 $p$ 个 $b_{si}$ 之和是可用上界；有额外约束时，忽略
约束所得的更松上界仍有效。裁剪下界不能随意改变式子方向或将真实 $J$ 的值混入代理剪枝。

## 7. 共同选择额外代价的区间

在同一可行域，设精确单项最优为 $m_s=\min_C F_s(C)$，定义：

$$
\Phi(C)=\sum_s\omega_sF_s(C),\quad
\Delta^*=\min_C\Phi(C)-\sum_s\omega_sm_s.
$$

取 $0<\underline Z_s\le Z_s(C)\le\bar Z_s$，令：

$$
\Gamma=\min_C\sum_s\omega_s
\frac{N_s(C)-m_sZ_s(C)}{\bar Z_s},\qquad
\eta=\min_s\frac{\underline Z_s}{\bar Z_s}>0.
$$

设 $C_\Gamma$ 是该线性问题的最优可行集合，则：

$$
\boxed{\Gamma\le\Delta^*\le
\Phi(C_\Gamma)-\sum_s\omega_sm_s\le\Gamma/\eta.}
$$

**推导。** 每一项满足：

$$
\frac{N_s-m_sZ_s}{\bar Z_s}
=\frac{Z_s}{\bar Z_s}(F_s-m_s),\quad
\eta\le Z_s/\bar Z_s\le1.
$$

非负加权后，线性残差值夹在 $\eta[\Phi(C)-\sum_s\omega_sm_s]$ 与
$\Phi(C)-\sum_s\omega_sm_s$ 之间，再分别代入共同最优集合与 $C_\Gamma$ 即得。

分母范围窄时该区间可能有用；范围宽时 $\eta$ 小，不能把区间称为紧界。它度量的是代理
共同选择代价，不是现实任务损失、搜索加速保证或通用“共享电路”程度。

### 一个可手算的例子

令 $k=1$，三组分别为 A/B/C，两个响应权重均为 1：

| 响应 | $N_0$ | $Z_0$ | $a_A,a_B,a_C$ | $b_A,b_B,b_C$ |
| --- | --- | --- | --- | --- |
| 1 | 5 | 10 | -2, 1, 0 | 1, -1, 2 |
| 2 | 5 | 10 | 1, -2, 0 | -1, 2, 1 |

$m_1=3/11,m_2=1/4$，共同最优选 C，$\Delta^*=23/66$。
取两个分母范围均为 $[9,12]$，则 $\eta=3/4$；线性残差选择 B，
$\Gamma=13/44$，$\Gamma/\eta=13/33$。所以：

$$
13/44\le23/66\le13/33.
$$

这里的数字是推导说明，不是采自模型的实验。

## 8. 精确重复响应与组耦合

相同图像、相同文字前缀，在问题分歧点之前的同一头/位置具有相同因果计算。
若确实是相同向量响应 $Y_a(C)$，可先合并读出：

$$
\sum_q\omega_qg_{qa}^\top Y_a(C)=
\left(\sum_q\omega_qg_{qa}\right)^\top Y_a(C).
$$

只有经结构确认的重复项这样合并；分母相近或样本相似不够。

对响应的分组 $\mathcal P$，还可用组内共同选择加强独立下界：

$$
LB_{\mathcal P}=\sum_{B\in\mathcal P}
\min_{C\in\mathcal F}\sum_{s\in B}\omega_sF_s(C).
$$

精确组内求解时，合并组不会使此界变松。若使用近似下界，保留已有有效界并取最大，
不能自动继承精确求解的单调性。一般二元分式和求解已有 [R12](02_related_work.md#r12)，
待实现的专用收益应来自缓存复用、便宜认证和搜索效果，而非问题重命名。

## 9. 让联合替换保留可解线性子问题

在一个可行的大小 $k$ 当前集合 $C^{(r)}$ 附近，允许最多换出/换入 $h$ 组：

$$
\mathcal F_r=\{C:|C|=k,\ |C\triangle C^{(r)}|\le2h\}
=\{C:|C|=k,\ |C\cap C^{(r)}|\ge k-h\}.
$$

对节点剩余 $p$ 个名额，把 $U$ 分为 $U_{\rm in}=U\cap C^{(r)}$ 与
$U_{\rm out}=U\setminus C^{(r)}$。枚举从内部取 $a$ 个，条件为：

$$
0\le a\le|U_{\rm in}|,\quad
0\le p-a\le|U_{\rm out}|,\quad
|I\cap C^{(r)}|+a\ge k-h.
$$

每个可行 $a$ 的线性最优是内部最小 $a$ 项加外部最小 $p-a$ 项；取其中最小。
因此单分式残差检验、线性界和候选构造仍可共用线性选择器。
这需要在真实实现中与小规模穷举核对。

## 10. 内外层搜索规格

**初始化**：用所有方法共享的未干预缓存和约定初始化，得到一个可行大小 $k$ 集合。
可先用未干预点梯度提案；上述信赖域只在已有可行当前集合后使用。

**内层**：固定当前真实下游梯度；建立有效下界、可行候选和代理上界；必要时刷新单项界、
分组或分支；返回候选、代理间隙、时间与终止原因。固定时间停止时保留当前可行解与有效界。

**外层**：通过原生 $D_q$ 计算候选的真实 $J$。若预测下降为正，定义：

$$
\rho_r=\frac{J(C^{(r)})-J(C_{\rm trial})}
{\widehat J_r(C^{(r)})-\widehat J_r(C_{\rm trial})}.
$$

根据真实改善接受候选，并用实际/预测一致性调整替换半径；在新点重算梯度。
若预测下降非正，不计算这个比值。具体阈值由验证集确定并在测试冻结。
还应报告 $\|\Delta Y\|$，因为少量槽位替换也可能引起大响应。

当 $h=1$ 仍无接受步，不能继续缩为 0 并宣布全局最优。按预声明规则检查替代提案、
扩大邻域/重启或以“局部停滞/预算停止”结束。真实可行候选可保留，但不据此剪掉代理分支。

在图像、问题、端口、donor 与固定 Q/V 不变时，外层重新线性化主要改变标量分子；
K 权重/分母统计可复用。任一前提改变则缓存重新建立。

## 11. 代理改善怎样连接真实行为

若在本轮可行域确有统一误差界：

$$
|J(C)-\widehat J_r(C)|\le\varepsilon_{\rm model}\quad\forall C\in\mathcal F_r,
$$

内层候选满足代理误差 $\varepsilon_{\rm solve}$，则：

$$
J(C_{\rm trial})-\min_{C\in\mathcal F_r}J(C)
\le\varepsilon_{\rm solve}+2\varepsilon_{\rm model}.
$$

证明是在真实最优集合与候选两端各支付一次模型误差，再加代理求解误差。
这个条件定理区分了两种难题；少量采样的最大误差不构成统一误差界。
实际研究要测量代理误差和排序/提案转化，不能把内层证书当成真实行为的最优性证明。

## 12. 收益、成本与不可省略的基线

总成本至少按下式分解：

$$
T=T_{\rm prefix}+T_{\rm stats}+T_{\rm search}
 +N_D T_D+N_{\rm backward}T_{\rm backward}.
$$

$T_D$ 按具体问题/批次实际计量，不假定所有下游调用同价。所有方法享有相同的无关计算
缓存收益；求解器的额外 CPU 成本、失败提案及重新线性化都计入总时间。

比较 random、AtP* 单项 top-k、真实条件贪心、贪心＋1-swap、连续 K 替换掩码优化，及
相同外层中使用可加代理的适配基线。小实例用穷举核对，用通用分式求解路线比较代理值与
间隙。详细验收见 [S01 协议](protocols/S01_Joint_K_Search.md)。

如果真实收益仅来自外层迭代或普通缓存，缩小专用分式求解的贡献表述；如果界很松或 CPU
分支成本吞掉 GPU 节省，继续依据实际瓶颈修改方法，而不宣称代数精确已经证明效率收益。
