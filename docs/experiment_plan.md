<!-- 提供新版已实现实验入口；科学树与首批预算只链接唯一来源。 -->
# 实验入口：统一子包、配置与协议

科学编号和依赖见 [06](research/06_experimental_design.md)，下一批执行清单见
[08](research/08_resources_and_reproducibility.md)。B03、M03–M09、V01、S01仅有计划，
不使用旧入口冒充新实验，也不创建占位实现。

## 已实现的入口

模块均以 `reladdr.experiments.` 开头：

| 实验 | 模块／子命令 | 配置示例 |
| --- | --- | --- |
| B01_Behavioral_Baseline 正式基线 | `b01_behavioral_baseline` | `B01_Behavioral_Baseline.yaml` |
| B01 工程检查 | `b01_behavioral_baseline.smoke` | `B01_Behavioral_Baseline__smoke.yaml` |
| B02_Binding_Replication | `b02_binding_replication` | `B02_Binding_Replication.yaml` |
| M01_Interface_Diagnostics | `m01_interface_diagnostics linear` / `grid` | `M01_Interface_Diagnostics__linear.yaml` / `__grid.yaml` |
| M02_Path_Diagnostics | `m02_path_diagnostics mediation` / `freeze` | `M02_Path_Diagnostics__mediation.yaml` / `__freeze.yaml` |

所有当前ID和协议版本按 [配置表](../configs/README.md) 使用新版；不保留旧模块、E编号配置别名或跳转协议。
已完成run仍保留原配置／版本，历史科学结果对应原执行代码，不宣称新版入口已做过GPU复验。

正式入口支持以下离线数据预检，不加载模型，会生成本地prepare-only run：

```sh
uv run --locked python -m reladdr.experiments.b01_behavioral_baseline configs/B01_Behavioral_Baseline.yaml --prepare-only
uv run --locked python -m reladdr.experiments.b02_binding_replication configs/B02_Binding_Replication.yaml --prepare-only
uv run --locked python -m reladdr.experiments.m01_interface_diagnostics linear configs/M01_Interface_Diagnostics__linear.yaml --prepare-only
uv run --locked python -m reladdr.experiments.m01_interface_diagnostics grid configs/M01_Interface_Diagnostics__grid.yaml --prepare-only
uv run --locked python -m reladdr.experiments.m02_path_diagnostics mediation configs/M02_Path_Diagnostics__mediation.yaml --prepare-only
uv run --locked python -m reladdr.experiments.m02_path_diagnostics freeze configs/M02_Path_Diagnostics__freeze.yaml --prepare-only
```

工程smoke只在 [GPU初调](gpu_smoke.md) 的独立授权下运行，不提供伪造的prepare-only结果。
参数校验与执行在各子包；共享三标签评分/移植在`_shared/patching.py`，原生端口仍由models/interventions集中维护。

## 当前停点

[B01旧 E01 discovery](reports/e01_discovery_2026-10-20.md) 与
[B02/M01/M02旧 E02/E03](reports/e02_e03_2026-10-20.md) 的指标和来源保持不变。
下一批先B03自然行为与M03端口验收，通过门槛后才做M04有限自然捕获。
后续M05/M06需要实际消费接口；地址中继与S01集合搜索均不阻塞主线。
