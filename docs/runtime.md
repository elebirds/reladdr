<!-- 说明公共运行层如何接入实验，以及本地记录的契约和使用方式。 -->
# 把实验计算接入可靠的本地记录

先由具体实验校验研究参数，再把计算放进 `RunContext`。运行层负责配置快照、状态、日志和
结果保存；实验负责样本、模型、比较、指标含义和结论。

## 先检查公共配置，不把它当完整实验校验

```sh
uv run --locked reladdr config check --common-only path/to/config.yaml
```

该命令只校验公共字段，保留尚未解释的 `experiment.parameters`，不创建 run。
运行前必须调用 `load_config(path, parameter_model=Parameters)`：`Parameters` 由实验定义，
必须声明 `extra="forbid"`，推荐继承 `reladdr.runtime.config.StrictModel`。
加载后的 `validation_scope` 为 `experiment`；仅通过公共检查的配置不能创建 `RunContext`。

配置公共字段见[配置规范](../configs/README.md)。YAML 安全加载，重复字段、未知公共字段和
错误类型报错。配置路径相对配置文件所在目录解析，实际输出路径写入快照。私有路径由实验
schema 转换后调用 `resolve_path(value, config.source.parent)`；公共层不猜测字段名。

## 实验持有自己的协议，调用公共接口记录结果

下面片段嵌入真实实验入口：`Parameters`、`compute` 及其结果格式由该实验提供。

```python
from reladdr.runtime import RunContext, load_config

config = load_config(config_path, parameter_model=Parameters)
with RunContext(config) as run:
    run.record_metadata("model", model_metadata)
    with run.time_phase("inference"):
        results = compute(config.parameters, rng=run.rng)
    run.artifacts.write_jsonl(
        "baseline/predictions.jsonl",
        results.rows,
        description="逐样本预测与真值",
        sync=False,
    )
    run.artifacts.write_json(
        "baseline/summary.json",
        results.summary,
        description="实验定义的汇总",
        sync=True,
    )
    for name, value in results.metrics.items():
        run.log_metric(name, value)
```

公共层不计算或自动补齐准确率、地址误差或求解间隙。指标接受有限数值，`step` 必须是非负
整数；耗时使用秒，其他指标由调用者声明 `unit`。没有测量的数值不填零。
`time_phase` 即使阶段失败也记录实际耗时。每次执行还记录 `runtime.wall_time_s`。

`run.rng` 是独立的 Python RNG，不改变全局随机状态。模型适配器接入时需要另行设置并记录
NumPy/PyTorch 等实际使用的 RNG 与确定性选项；一个 seed 不承诺所有后端逐位一致。

## 一次执行对应一个不覆盖旧结果的目录

```text
runs/<run-id>/
├── config.resolved.json      # 实际配置，含私有 schema 展开的默认值
├── metadata.json             # 身份、seed、代码、依赖与实验补充信息
├── status.json               # 执行状态和实际耗时
├── metrics.jsonl             # 按写入顺序保存指标，不只保留最后一个值
├── manifest.json             # 已完成产物及显式同步标记
├── tracking.json             # 独立的同步状态
├── run.log                   # 该 run 的日志，不改全局 logging 配置
└── artifacts/                # 实验自定子目录和格式
```

run ID 自动包含 UTC 时间、标签和随机后缀；可显式指定，但冲突时直接报错。
同一个上下文只能执行一次。输出根目录相对配置文件解析，而不是相对启动时的工作目录。

metadata 公共部分不允许实验覆盖。`record_metadata(section, values)` 在 `details` 下新增节，
每节只能写入一次；实验需在这里记录模型/processor revision、数据版本、split、实际
预处理、dtype、attention backend、端口和 donor 等。当前公共层不会伪造 GPU 环境信息。

代码来源取自项目源码所在 checkout，不取配置文件所在仓库。保存 commit、dirty 列表、
相关源码差异与未跟踪文本源码快照；差异和源码快照默认只留本地。不在 Git checkout 中时
明确记录来源不可用；wheel 安装不会借用其所在目录的 Git 身份。正式结论应从已提交版本
运行。不会收集完整环境变量或任意命令行。

## 失败和中断也是有效运行记录

- 正常退出：`completed`；异常：`failed`，保留原异常并继续抛出。
- `KeyboardInterrupt`、`SystemExit`：`interrupted`。
- `SIGKILL`、默认 `SIGTERM`、断电或磁盘写入失败可能来不及收尾；保留 `running` 或已写入
  的文件，不推断成功。查看命令不修改状态，也不恢复执行。
- 结束后关闭专属日志句柄，禁止通过上下文继续写指标或产物。

```sh
uv run --locked reladdr run inspect path/to/run
uv run --locked reladdr run inspect --json path/to/run
```

状态 JSON 使用同目录临时文件和原子替换；指标按条追加。每个 run 只支持一个写入者。
本轮没有多进程共享记录器、任务调度或恢复计算功能。

## 产物显式登记，不扫描整个目录上传

`write_json`、`write_text`、`write_jsonl` 不覆盖已有文件，成功后登记到 manifest。
逐条 JSONL 写入失败时，未完成文件留在本地但不登记、不上传。
绘图库可先取得 `path = run.artifacts.path_for("plots/figure.svg")`，写完后调用
`register("plots/figure.svg", description="…", sync=True)`。登记后不要再改写文件。

只允许 run 内路径，拒绝父目录跳转和指向外部的符号链接。默认 `sync=False`；大激活、
数据和权重不默认同步。JSON 禁止 NaN/Infinity，不自动把任意 Python 对象转成字符串。

## 凭据不进入配置或记录

服务地址与认证通过批准的环境变量提供，不支持把 `tracking_uri`、密码、API key 等写入
公共或私有配置。敏感字段报错时不回显值；日志、异常消息和 JSON 会遮盖明确认证环境变量
中的密码/令牌实际值；用户名不作为任意文本遮盖词，避免损坏正常路径。不会加载 `.env`
或转储整个环境。实验仍须避免记录其他未声明的私有内容。

## 参数/指标和文件归档分别开关

初期建议只同步参数/指标，不让 OSS 配置阻塞实验：

```yaml
# 只启用 MLflow 参数/指标；所有文件继续完整保存在本地。
tracking:
  mode: mlflow
  experiment_name: reladdr-E01
  params_metrics: true
  artifacts: "off"
```

`mode: local` 是总开关：不导入 MLflow、不访问服务，CLI 也不能借覆盖选项上传这类历史 run。
`mode: mlflow` 时，`params_metrics` 控制参数/指标 API；`artifacts` 控制文件归档：

| 文件归档 | 文件怎样上传 | 客户端需要什么 |
| --- | --- | --- |
| `"off"` | 不检查产物文件、不上传文件 | 仅同步参数/指标时只需 Tracking 服务地址与认证 |
| `direct` | MLflow SDK 按远程 `s3://` 地址直传 OSS/S3 | `tracking`、`oss` 可选依赖及 endpoint、存储凭据 |
| `proxy` | MLflow SDK 经服务端产物代理转存 | `tracking` 可选依赖与服务认证；客户端不需要 OSS 凭据 |

两项都关闭时不需要服务地址或 SDK，也不创建远程 run。关闭开关不会删除已上传内容。
YAML 中 `"off"` 必须加引号：当前安全加载器会把未加引号的 `off` 解释成布尔值，严格校验会拒绝。

为兼容已有配置，省略 `artifacts` 或写 `null` 时保留原先 SDK 原生路由与文件同步行为，
状态中记为 `native`；它不是新模式选项。新配置请明确填写上述三种模式，不靠省略关闭归档。

## 结束后显式同步，必要时在原 run 上改策略

仅同步已经结束、配置中启用 MLflow 的 run。服务地址和认证通过批准机制注入，
`MLFLOW_TRACKING_URI` 必须带 `https://` 等协议，不能含 userinfo、查询或片段。
应用不会自动加载 `.env`；需要使用本地忽略的 `.env` 时，通过 uv 显式注入：

```sh
uv run --locked --extra tracking --env-file .env reladdr tracking sync path/to/run \
  --params-metrics --artifacts off
```

CLI 覆盖只写入 `tracking.json`，不改原配置快照、不重跑计算。之后裸重试沿用上次策略，
不会在一次关闭归档后又意外启用旧配置。需要补传文件时仍使用同一 run：

```sh
# 已配置好直连的客户端依赖、endpoint 与凭据后，只补传文件。
uv run --locked --extra tracking --extra oss --env-file .env reladdr tracking sync path/to/run \
  --no-params-metrics --artifacts direct

# 远程实验已使用代理产物地址时，改走代理归档。
uv run --locked --extra tracking --env-file .env reladdr tracking sync path/to/run \
  --no-params-metrics --artifacts proxy
```

模式必须与该远程 run 的产物地址一致：`direct` 要求 `s3://`，`proxy` 要求
`mlflow-artifacts:/` 或 MLflow HTTP 产物代理端点。客户端会核对，不自动回退或修改服务器。
已有直连实验不会因服务器后来启用代理而自动迁移；应使用正确产物配置的实验。
参见 [MLflow 的直连与代理配置](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/)。

## 同步哪些内容

参数/指标同步保持原有契约：上传 `experiment.id`、`experiment.protocol_version`、`run.seed`、
`config.schema_version` 四项公共参数，以及全部指标历史。完整实验私有参数仍保存在
`config.resolved.json`，不猜测如何展开成 MLflow 参数表；关闭归档后，这份完整配置只在本地。

启用文件归档时，上传配置、metadata、执行状态、指标和 manifest 五份记录文件，以及
manifest 里 `sync=True` 的产物。日志、源码快照和大激活不默认上传；单文件限制 10 MiB。
检查登记后的文件是否改变、缺失、越界或超限，不符合时明确报错。新登记的 `sync=True`
文件记录内容摘要，防止同长度、同时间戳的改写漏检；不扫描或校验整份模型/数据目录。
旧清单继续使用原来的尺寸/时间戳依据，不把它说成已有内容完整性保证。

两个开关控制不同传输渠道，不会改写文件内容：即使 `params_metrics: false`，启用归档后
配置和指标文件中仍包含原始参数/指标。关闭归档时不检查产物文件，也不导入存储客户端。

## 两类进度分别保存，失败后原位重试

`tracking.json` 保存本次策略、总体状态和 `params_metrics_status`、`artifacts_status`，
并分别保存参数确认标记、指标游标、已上传文件列表。两类子状态使用
`pending/disabled/syncing/synced/failed`；查看命令会展示实际启用的策略。

- 参数/指标先同步；显式归档失败不会抹去已确认的参数与指标，下次只补未完成部分。
- 若认证或参数/指标请求失败，停止本轮同步，不转向对象存储尝试绕过失败。
- 可以把失败的归档切成 `off`，先完成参数/指标；之后重新启用时保留文件续传进度。
- 复用远程 run ID；创建响应丢失时按稳定 UID 找回，不盲目新建。
- 已绑定的 run 不静默换服务，远程身份不符时拒绝继续，进程锁防止并发同步。
- 旧的原生路由继续在远程写入前检查全部上传文件，保留原先调用契约。

同步失败不改变 `status.json` 的实验执行状态。选定渠道全部完成后，才收尾远程 run；
远程最终状态按本地 `completed/failed/interrupted` 映射。总体 `synced` 表示选定渠道完成，
不表示关闭的归档已上传。CLI stdout 始终为 JSON，SDK 的提示输出转到 stderr。

指标提交采用“至少一次”语义：若服务器已接收但响应丢失，重试可能产生相同时间戳/step
的远程历史重复；已确认的指标和文件会跳过，本地原始指标不重复。

## 已验证到哪一步

CPU 测试覆盖独立开关组合、路由核对、故障隔离、续传和旧配置兼容。真实 MLflow SDK 在
临时文件后端验证了完整上传/读回，以及关闭归档后仅同步参数/指标、不上传文件。

2026-10-19 已在实际 Tracking 服务复用专用基础设施 run，完成参数/指标读回、终态检查和
关闭归档后的重复同步，未重复创建 run 或写入已确认指标；没有联系 OSS。
真实 OSS 直连/代理的文件上传与下载尚未验证，需要相应客户端存储配置或服务端代理实验。
实际接入的地址曾为裸主机名，该次检查仅在进程内补齐 HTTPS，没有修改 `.env`；正常命令
仍要求提供完整 URI。本地实测回执留在对应忽略的 run 目录，不将认证或机器信息写入仓库。

实现参考：[Pydantic 模型](https://docs.pydantic.dev/latest/concepts/models/)、
[PyYAML 安全加载](https://pyyaml.org/wiki/PyYAMLDocumentation)、
[MLflow 客户端接口](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html)、
[MLflow 产物存储](https://mlflow.org/docs/latest/self-hosting/architecture/artifact-store/)。
