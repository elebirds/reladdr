<!-- 介绍 RelAddr 的研究目标、当前实现、开发入口和文档导航。 -->
# RelAddr

**Relation-Conditioned Visual Addressing · 视觉语言模型的关系条件寻址**

研究视觉语言模型如何把参照对象与关系条件接入目标选址，再读取目标属性。
机制研究与联合干预集合搜索是两条独立研究线，共享模型适配和记录基础设施。

公共运行层已实现：配置校验、本地运行生命周期、日志、指标、产物与来源记录，以及可选
MLflow 显式同步和重试。参数/指标独立开关，文件归档可选关闭、OSS 直连或服务端代理。
已加入三形状场景、Qwen2-VL BF16 推理与 E01 工程冒烟入口。
正式网格/留出协议、激活干预和搜索算法尚未实现；初调结果与研究结论分别记录。

## 开始开发

```sh
uv sync --locked
uv run --locked reladdr --help
uv run --locked pytest
```

项目固定 Python 3.12.14；依赖由 `pyproject.toml` 和 `uv.lock` 共同管理。
使用项目 `.venv`，不通过全局 pip 安装依赖。基础运行不依赖 GPU 或 MLflow。

## 查看配置和结果

```sh
uv run --locked reladdr config check --common-only path/to/config.yaml
uv run --locked reladdr run inspect path/to/run
uv run --locked reladdr run inspect --json path/to/run
```

公共检查明确输出 `common_only`，不代表实验参数已通过校验。
真实实验入口需要提供自己的参数模型，再通过 `RunContext` 执行。
三形状初调入口为 `python -m reladdr.experiments.e01`，见[GPU 初调](docs/gpu_smoke.md)。

已结束且配置中启用 MLflow 的 run，可以在注入批准的服务地址与认证后显式同步：

```sh
uv run --locked --extra tracking reladdr tracking sync path/to/run --params-metrics --artifacts off
```

初期建议关闭文件归档。真实 SDK 已通过临时本地后端的完整读回检查；实际 Tracking 服务的
参数/指标与重复同步也已验证。真实 OSS 文件链路尚未验证。同步不重跑实验，具体配置见
[运行层用法](docs/runtime.md#参数指标和文件归档分别开关)。

## 公共能力与实验协议分开

- `reladdr.runtime`：公共配置、执行状态、本地记录与产物管理。
- `reladdr.experiments`：后续实验私有配置、比较协议、执行和分析；随真实实现加入。
- `reladdr.tasks`：场景、问题真值与场景族划分。
- `reladdr.models`：模型加载、token 映射与原生张量接口。
- `reladdr.interventions`：缓存、替换、重算与恢复。
- `reladdr.mechanisms`：地址表示、关系算子与原生路径分析。
- `reladdr.evaluation`：可复用的研究评价与统计。
- `reladdr.search`：联合干预集合搜索。

场景与模型组件已有初调实现，其余组件仍是说明。实验私有协议留在 `reladdr.experiments`，
不建立统一实验评分或庞大的实验基类。

## 文档

- [GPU 初调](docs/gpu_smoke.md)：Instruct、三形状任务、图像对照和真实输出。
- [运行层用法](docs/runtime.md)：配置边界、Python 接口、输出布局和状态含义。
- [开发与复现](docs/development.md)：依赖、检查、CPU/GPU 分工与记录约定。
- [配置规范](configs/README.md) · [测试说明](tests/README.md) · [贡献约定](AGENTS.md)。
- [研究档案](docs/research/README.md)：背景、文献、原生接口、机制假设、数学推导与实验设计。
- [实验实施入口](docs/experiment_plan.md)：下一步 E01，随后分别推进机制与搜索研究。
