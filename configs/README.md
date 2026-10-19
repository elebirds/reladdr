<!-- 说明公共配置结构，以及由各实验独立维护的研究参数约束。 -->
# 公共运行配置与实验参数分开

当前没有 E01 等实验的执行入口，因此不提供假装可运行的实验 YAML。
公共格式如下；`parameters` 的具体内容由对应实验定义。

```yaml
# 说明一次执行的身份、公共运行设置与实验私有参数。
schema_version: 1
experiment:
  id: E01
  protocol_version: "1"
  parameters: {}
run:
  seed: 19
  output_root: ../runs
  label: baseline
  log_level: INFO
tracking:
  mode: local
```

- `schema_version` 当前为 1；实验 ID 为字母/数字/下划线/短横线组成的标识。
- `protocol_version` 显式填写字符串；`seed` 是 0 至 `2**32-1` 的整数，不接受字符串或布尔值。
- `output_root` 相对该 YAML 的目录解析，默认 `runs`。`label` 可省略。
- `tracking.mode` 为 `local` 或 `mlflow`；后者必须提供非空 `experiment_name`。
- 服务地址和凭据不放在配置中。YAML 不展开环境变量、不继承其他配置、不接受重复字段。

公共 CLI 只检查这层结构，并明确标注未校验私有参数：

```sh
uv run --locked reladdr config check --common-only path/to/config.yaml
```

真实入口调用 `load_config(path, parameter_model=该实验的参数模型)`；私有模型必须拒绝未知
字段。E03 不需要 E04 的参数，CPU 工具也不必填写模型路径。私有路径由实验显式解析。

## 研究参数必须满足各自的实验协议

模型实现后需明确固定 checkpoint/processor revision、dtype、attention backend、视觉 token
预算和实际预处理。数据需明确生成器/数据版本、关系语义、split 与场景分组；同图多问及
镜像/换色变体共享 scene-family ID，不跨 discovery/validation/test。

机制实验明确层/头/读取位置、拟合维度及选择参数的 split；搜索实验另行明确原生端口、
pre/post-RoPE、候选组、固定 donor、k、问题权重/foil、停止规则与真实调用/时间预算。
这些约束由各自 schema 和实验验证，不转嫁给公共运行层。
