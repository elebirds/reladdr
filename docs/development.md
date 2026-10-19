# 开发与复现

## 先恢复一个不依赖 GPU 的开发环境

在仓库根目录执行：

```sh
uv sync --locked
uv run --locked python -c "import reladdr; print(reladdr.__file__)"
uv run --locked ruff check .
uv run --locked ruff format --check .
uv build
```

项目固定 Python 3.12.14；`pyproject.toml` 允许 Python 3.12 系列。
默认安装空项目包以及 pytest、ruff。基础运行依赖暂为空：实际任务和模型尚未实现。
环境位于项目 `.venv`。入口统一使用 `uv run --locked`，避免调用全局环境中的同名库。

## 修改依赖时同时更新锁文件

开发工具用 `uv add --dev`，运行依赖用 `uv add`，可选客户端用 `uv add --optional tracking`。
依赖声明和解析后的 `uv.lock` 同次提交。普通复现不执行全量升级。
CI 固定 uv 0.12.10；本地若升级构建工具，应核对 CI 与构建后端并运行相同检查。

[uv 的锁定语义](https://docs.astral.sh/uv/concepts/projects/sync/) 中，`--locked`
会在声明与锁文件不一致时报错，而不是悄悄更新锁文件。

## Mac 开发与 Linux GPU 运行分工

Mac 先承担场景设计、文档、CPU 测试与结果分析。Qwen2-VL-7B-Instruct 是首个计划分析对象，
本轮尚未安装 PyTorch、Transformers、FlashAttention 或任何模型权重。

接入模型时，再声明 `gpu` 可选依赖并固定实际验证过的 PyTorch/Transformers 组合。
Linux CUDA 构建的来源必须显式配置；不能依据 Mac 上的包导入推断 CUDA 可用。
uv 锁文件不能固定显卡驱动，后续运行仍需记录驱动、CUDA、GPU、dtype、attention backend。
不同 attention 后端或量化精度可能改变被解释的计算，比较时必须明确。

[uv 的 PyTorch 指南](https://docs.astral.sh/uv/guides/integration/pytorch/) 说明了按平台和
accelerator 配置来源的方法。当前不盲选一个 CUDA wheel，也不为导入空包安装 GPU 栈。

## MLflow 只准备客户端

```sh
uv sync --locked --extra tracking
uv run --locked --extra tracking python -c "import mlflow; print(mlflow.__version__)"
```

`tracking` 使用 [mlflow-skinny](https://pypi.org/project/mlflow-skinny/)；无需在 GPU 机器部署
服务端。上述命令不连接任何 MLflow 服务。`.env.example` 只列变量，不自动加载；认证
通过已批准的 shell/secret 机制注入，不将实际值写入仓库、配置快照或日志。

后续接入要求：一次执行对应一个 run；同一研究问题的多次运行按 experiment 组织。
显式记录参数和指标，不默认启用 autolog。只上传选定的小型结果表、配置和图；不默认
上传模型、数据集、全量激活。启用前核对客户端版本与服务，验证创建/记录/上传/读回。
本轮没有连接服务或创建远程 run。

本地记录先完整落盘。同步失败要在该 run 中明确标记，不把未上传说成已同步，也不丢掉
已完成的 GPU 结果。当前没有实现记录器、自动重试队列或同步命令。

## 运行记录的接口约定

后续每次执行产生独立的 `runs/<run-id>/`：实际配置、metadata、汇总 metrics、逐样本结果、
干预明细、日志与关键图。这个布局是约定，当前尚未实现 runner。

Metadata 至少包含代码 commit/相关未提交改动、模型与 processor revision、数据/生成器版本、
split 与 scene-family ID、seed、精度、后端、实际 token 长度、干预接口、固定 donor、设备环境。
不记录整个进程环境，不收集凭据。

成本以明确单位保存：wall_time_s、阶段耗时秒数、peak_allocated_bytes、peak_reserved_bytes、
forward_calls、backward_calls。下游调用数和包含 CPU 求解的总时间分别比较，不把一次反向
当作一次前向。具体字段随首个实现确定并同步更新配置说明。

Git 管理代码、配置与小型研究结论。数据、权重和大体积激活使用独立存储；关键原始产物
需要备份，目录清单不能替代备份。研究解释链接 run ID，不手工维护第二份指标表。

## 当前开发检查能证明什么

CI 配置覆盖 Linux 上锁文件恢复、Ruff、所有空子包及可选客户端导入、sdist/wheel 构建。
本地验收运行相同类别的检查。没有业务测试或模型实验；首次 GitHub 执行要在推送后确认。
有真实测试后再加入 pytest 步骤，不通过占位测试制造“已验证”的状态。
