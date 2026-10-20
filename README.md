<!-- 介绍项目问题、已实现能力、统一入口与研究文档的位置。 -->
# RelAddr

**Relation-Conditioned Visual Addressing · 视觉语言模型的关系条件寻址**

研究图像依赖的参照绑定怎样与关系形成选择条件，并驱动目标属性读取。
机制解释与联合干预集合搜索独立验收，共享模型适配和运行记录。

已有行为基线、位置／属性阶段模式复现、动态参照接口诊断和后缀恢复／旧写入诊断；
尚未恢复具体关系算法。新实验树及第一批计划已写入协议，未启动模型实验。

## 从哪里开始

- [实验登记与依赖树](docs/research/06_experimental_design.md)：`编号_English_Name`、状态与历史证据映射。
- [第一批执行清单和预算](docs/research/08_resources_and_reproducibility.md)：准备、门槛和停止条件。
- [原生接口](docs/research/03_interfaces_and_notation.md) · [机制假说](docs/research/04_mechanism_hypotheses.md)。
- [当前证据与决策](docs/research/07_outcomes_and_decisions.md) · [完整研究导航](docs/research/README.md)。

## 开发与离线检查

```sh
uv sync --locked
uv run --locked reladdr --help
uv run --locked pytest
```

Python固定3.12.14，依赖由`pyproject.toml`和`uv.lock`共同管理；GPU和MLflow是可选依赖。
本地CPU检查不加载模型、不下载权重、不访问远程服务。环境和检查见 [开发说明](docs/development.md)。

## 已实现的实验子包

- `b01_behavioral_baseline`：正式行为基线；同包`smoke`子模块仅作工程初调。
- `b02_binding_replication`：位置／属性阶段模式复现。
- `m01_interface_diagnostics`：`linear`和`grid`接口诊断。
- `m02_path_diagnostics`：`mediation`和`freeze`诊断。

模块前缀为`reladdr.experiments.`，只使用新版入口与ID。历史run和来源快照不改写；
未实现的计划不创建空子包或占位配置。完整命令和协议见 [实验入口](docs/experiment_plan.md)。

例如，已有协议的数据预检不会加载模型，但会创建本地prepare-only run：

```sh
uv run --locked python -m reladdr.experiments.b02_binding_replication configs/B02_Binding_Replication.yaml --prepare-only
uv run --locked python -m reladdr.experiments.m01_interface_diagnostics grid configs/M01_Interface_Diagnostics__grid.yaml --prepare-only
```

这不是下一批B03/M03的执行命令；它们的专用实现仍待后续授权开发。配置模板的模型来源需要执行前核实，
当前操作不授权GPU运行、下载或租用资源。

## 查看记录

```sh
uv run --locked reladdr config check --common-only path/to/config.yaml
uv run --locked reladdr run inspect --json path/to/run
```

公共配置检查不代表实验参数通过；正式入口另校验私有schema。运行层已有配置校验、状态、日志、
产物和来源记录，以及显式可选MLflow同步。参数／指标与文件上传独立开关，当前模板关闭远程同步。
真实Tracking参数／指标曾验证；文件归档链路尚未验证。使用见 [运行层](docs/runtime.md)。

## 模块边界与证据

实验私有执行在命名子包，共享移植流程在`experiments/_shared`；图像真值在tasks，
模型/GQA/mRoPE在models，替换与恢复在interventions，统计在evaluation。
机制解释与集合搜索未实现部分分别留在机制协议与搜索规格，不用统一实验基类制造空壳。

- [E01历史报告](docs/reports/e01_discovery_2026-10-20.md)：8族discovery、6,336次真实预测。
- [E02/E03历史报告](docs/reports/e02_e03_2026-10-20.md)：14个run、12,840次前向及解释边界。
- [GPU工程初调](docs/gpu_smoke.md) · [配置](configs/README.md) · [测试](tests/README.md) · [开发约定](AGENTS.md)。
