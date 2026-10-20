<!-- 区分已实现的运行层测试和后续研究组件需要补充的验收。 -->
# 测试真实契约，不用占位测试制造完成状态

```sh
uv run --locked pytest
```

当前测试覆盖公共配置与私有 schema 边界、本地生命周期、异常与中断、目录冲突、日志与
RNG 隔离、产物登记/路径约束、Git 来源和独立进程 CLI。还覆盖可选同步的故障、进度、
身份、上传白名单、并发锁和创建结果不确定时的恢复；包括参数/指标与 off/direct/proxy
归档的独立组合、路由不匹配、关闭归档后不检查文件、策略保留和分别续传。
测试使用临时目录，不访问服务、不下载模型。测试上下文的参数 schema 仅用于检查记录能力，不代表科学实验或模型结果。

## 可选 SDK 的真实离线闭环

```sh
uv run --locked --extra tracking pytest tests/runtime/test_tracking_sdk.py
```

真实 MLflow SDK 使用临时本地文件后端，从 CLI 同步到参数/指标/产物读回；另验证仅同步
参数/指标时即使登记文件缺失也不检查、不上传，并且重复同步不增加已确认的指标记录。
未安装可选依赖时明确跳过该项，其他运行层测试仍执行。不以本地后端通过代替实际服务器接入验证。

## B01_Behavioral_Baseline：三形状工程检查

`tests/tasks/test_synthetic.py` 检查渲染、对象/关系真值、确定性与场景族标识；
`tests/experiments/b01_behavioral_baseline/test_smoke.py` 用受控适配器检查记录闭环、空白图无准确率、错配图实际标签、
固定 foil 和失败记录。受控适配器结果不作为模型表现；真实 GPU 前向单独保存运行记录。

## B01_Behavioral_Baseline：正式行为基线

`tests/tasks/test_relational.py` 检查稳定场景族划分、逐方向组合留出、参照不变的反事实真值、
边界和模板顺序；`tests/experiments/b01_behavioral_baseline/test_formal.py` 检查 prepare-only 不加载模型、
未请求 split 不推理、场景族区间、配对、逐文件模型来源及故障后的已写输出保存。
受控适配器只验证执行契约，真实 GPU 行为另看实际 run。

## B02_Binding_Replication、M01_Interface_Diagnostics 与 M02_Path_Diagnostics

新增检查覆盖 Squares 几何/调色板、clean/donor 三标签可辨识性、图像束跨族/划分去重、
同指令参照移位、仅按真值的反向筛选、真实 token 词组匹配、临时钩子异常清理和后缀分组。
有 Torch 的环境另外检查张量替换/中介混合只改声明行列；Mac 不为此安装 GPU 栈。
实际 GPU 还核对原有生成与原生读出一致、逐层自身替换、完整 donor 状态恢复和钩子移除后恢复。

## 随研究组件加入的检查

1. **场景和问题**：目标唯一、方向一致、边界/空位规则、答案和标注一致。
2. **数据划分**：同场景及反事实变体不跨 discovery/validation/test。
3. **模型映射**：视觉 token 对应区域、GQA 分组与位置编号一致。
4. **干预语义**：no-op 不改输出，异常后恢复，只改变声明端口。
5. **解析响应**：小型数值实例及原生接口中解析与重算一致。
6. **集合求解**：小规模穷举核对代理、可行性、界和间隙，行为指标独立计算。

真实模型/GPU 检查与 CPU 测试分开；CI 不为导入空研究包临时下载模型。

## 新版入口契约

实验测试按 B01/B02/M01/M02 对应小写子包归组；入口测试检查canonical ID和版本校验、
非法身份不建run、原科学数据与元数据记录，以及真实CLI help／prepare-only。旧模块和ID不提供兼容。
prepare-only在临时目录内验证，不加载模型；smoke保持独立的工程用途。
