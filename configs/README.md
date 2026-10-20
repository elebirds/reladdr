<!-- 定义新版配置命名和参数边界；历史运行的配置副本只在来源记录保留。 -->
# 配置按实验名称组织

当前配置使用 `编号_English_Name.yaml`；同一实验的 case 用 `__case` 后缀，私有本地副本用
`.local.yaml`。例如 `M01_Interface_Diagnostics__grid.local.yaml`。
`experiment.id` 始终是完整实验名，case 不加入 ID；协议版本明确指定语义。

| 实验 | 配置前缀／case | 协议版本 | Python 入口 |
| --- | --- | --- | --- |
| B01_Behavioral_Baseline | 同名正式配置；`__smoke` 工程检查 | `b01-v1` / `b01-engineering-v1` | `b01_behavioral_baseline` / `b01_behavioral_baseline.smoke` |
| B02_Binding_Replication | 默认原句；`__adjacent`、`__frozen` | `b02-directional-v1` / `b02-adjacent-v1` | `b02_binding_replication` |
| M01_Interface_Diagnostics | `__linear`、`__directional`、`__reverse`、`__suffix`、`__linear_frozen` | `m01-adjacent-v1`；directional case 用 `m01-directional-v1` | `m01_interface_diagnostics linear` |
| M01_Interface_Diagnostics | `__grid`、`__grid_frozen` | `m01-grid-v1` | `m01_interface_diagnostics grid` |
| M02_Path_Diagnostics | `__mediation`、`__mediation_frozen` | `m02-suffix-mediation-v1` | `m02_path_diagnostics mediation` |
| M02_Path_Diagnostics | `__freeze`、`__freeze_frozen` | `m02-conditional-freeze-v1` | `m02_path_diagnostics freeze` |

模块前缀为 `reladdr.experiments.`。`__frozen` 只执行已经声明的比较，不自动选层。
上述当前配置和模块不提供 E 编号兼容入口。已完成run中的旧配置、版本与数据意义保持原样；
[映射表](../docs/research/06_experimental_design.md)仅用于读懂历史证据。
B03、M03–M09、V01、S01 尚无执行实现，不建立可运行样式的占位配置。

已有正式入口可用 `--prepare-only` 生成数据与预览而不加载模型。命令在
[实验入口](../docs/experiment_plan.md)。smoke 是真实模型工程检查，没有伪造的 prepare-only 路径。
模板里的 `unverified-local-checkpoint` 不是已验证来源；执行前在忽略的本地配置中填写固定模型与processor来源。

## 公共配置与实验私有参数分开

```yaml
# 公共结构示意；实际 parameters 使用对应已实现配置，不是可运行的空实验。
schema_version: 1
experiment:
  id: B02_Binding_Replication
  protocol_version: b02-directional-v1
  parameters: {}
run:
  seed: 19
  output_root: ../runs
  label: b02-binding-replication
  log_level: INFO
tracking:
  mode: local
  params_metrics: false
  artifacts: "off"
```

- `schema_version` 为1；公共层接受合法标识字符串，具体实验入口校验完整实验名和版本。
- `protocol_version` 显式字符串；seed是0至`2**32-1`的整数，不接受布尔或字符串。
- `output_root` 相对该 YAML 目录解析；配置不展开环境变量、不继承其他配置、不接受重复字段。
- `parameters` 由各实验私有schema验证，不把搜索线字段强加给机制线。
- `tracking.mode` 为local或mlflow；后者需非空experiment_name。凭据不进入配置。
- `params_metrics` 是严格布尔值；`artifacts` 独立选择 `"off"`、direct或proxy，仍受mode总开关控制。
  当前模板显式关闭上传；`off`要加引号。公共运行层仍保留缺省/null的旧SDK语义，不能误称为关闭。

公共CLI只检查公共结构，并明确标注未校验私有参数：

```sh
uv run --locked reladdr config check --common-only path/to/config.yaml
```

## 复现与授权

真实入口读取自己的参数模型。模型/processor revision、dtype、backend、视觉token、生成器版本、
scene-family和split必须记录。拟合与选择不能使用冻结测试标签。干预另记物理层/头/token、
pre/post-mRoPE、donor与重算范围；搜索另记候选集合、k、foil和预算。

同步使用 [运行层显式命令](../docs/runtime.md#参数指标和文件归档分别开关)，不自动上传数据或激活。
直连/代理必须匹配服务端run artifact URI，不静默切换。GPU执行、下载、租用和远程修改另行授权。
