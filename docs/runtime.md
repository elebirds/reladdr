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

## 本地运行结束后，再显式同步 MLflow

配置中设置 `tracking.mode: mlflow` 和 `tracking.experiment_name`，并通过批准机制注入
`MLFLOW_TRACKING_URI` 与所需认证，再执行：

```sh
uv run --locked --extra tracking reladdr tracking sync path/to/run
```

默认本地运行既不导入 MLflow，也不访问服务。只同步已经结束的 run；配置为 `local` 的
历史 run 不会因环境中存在服务器地址而被自动上传。服务地址不能含 userinfo、查询或片段。

同步包含公共参数、指标历史、配置/metadata/执行状态/指标/manifest 文件，以及 manifest
里 `sync=True` 的产物。完整私有参数保存在配置产物，不自动展开成巨大的 MLflow 参数表。
日志、源码快照和大激活不默认上传；单文件限制 10 MiB，超限明确报错。登记后改变的文件、
外部符号链接和缺失文件会在远程写入前拒绝。

`tracking.json` 独立保存 `pending/syncing/synced/failed`、远程 run ID、已确认指标游标和
已上传文件。同步失败不改变 `status.json` 的研究执行状态。再次执行同一命令重试：

- 复用远程 run ID；创建响应丢失时，按稳定 UID 找回原 run。
- 跳过已经确认的指标和文件，不重跑实验。
- 已绑定的 run 不静默切换服务器；远程身份不符时拒绝继续。
- 进程锁防止同一个 run 并发同步；已完全同步的 run 不重复上传。

指标提交采用“至少一次”语义：若服务器已接收指标但响应丢失，重试可能在远程历史中出现
相同时间戳/step 的重复记录，本地原始指标不重复。同步中断保留游标，下次可继续。
远程最终状态按本地 `completed/failed/interrupted` 映射，不把失败实验标成成功。

受控故障测试覆盖重试和隔离；真实 SDK 在临时文件后端完成创建、记录、上传与读回。
文件后端只在测试中临时显式启用，不作为实际服务部署方案。真实服务器的认证、兼容性和
读回需在首次实际接入时验证。

实现参考：[Pydantic 模型](https://docs.pydantic.dev/latest/concepts/models/)、
[PyYAML 安全加载](https://pyyaml.org/wiki/PyYAMLDocumentation)、
[MLflow 客户端接口](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html)。
