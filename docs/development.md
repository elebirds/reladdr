<!-- 说明 RelAddr 的 uv 工作流、实际检查及模型/记录依赖边界。 -->
# 开发与复现

## 恢复 CPU 开发环境并运行真实测试

```sh
uv sync --locked
uv run --locked reladdr --help
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
uv build
```

项目固定 Python 3.12.14；环境位于项目 `.venv`。基础运行依赖 Pydantic 与 PyYAML，开发依赖
pytest 与 Ruff。配置、生命周期、日志、产物、来源和 CLI 均有真实 CPU 测试；研究组件尚未
实现。测试不下载模型、不访问远程服务，也不把缺少模型实验当作模型检查通过。

## 修改依赖时一起更新锁文件

开发工具用 `uv add --dev`，运行依赖用 `uv add`，可选客户端用 `uv add --optional tracking`，
OSS 直连依赖用 `uv add --optional oss`。
依赖声明与 `uv.lock` 同次提交。普通复现不全量升级；使用 `uv run --locked`。
CI 固定 uv 0.12.10；更换构建工具时同步检查 CI 与构建后端。

[uv 锁定语义](https://docs.astral.sh/uv/concepts/projects/sync/) 中，`--locked` 在声明与
锁文件不一致时失败，不会悄悄更新锁文件。

## Mac 做开发，GPU 环境接入时再固定模型栈

Qwen2-VL-7B-Instruct 是首个计划对象。本轮未安装 PyTorch、Transformers、FlashAttention，
未下载模型或数据；导入公共运行层不导入 MLflow 或 GPU 栈。

接入时再声明 `gpu` 可选依赖，固定验证过的版本与 Linux CUDA wheel 来源。uv 锁文件不固定
显卡驱动，实际运行仍要记录 CUDA/驱动/GPU、dtype、attention backend 和 token 长度。
不同精度或后端可能改变被解释的计算，不能默认量化来规避显存要求。

[uv PyTorch 指南](https://docs.astral.sh/uv/guides/integration/pytorch/) 介绍了来源配置。
模型适配器独占 GQA、位置编码、token 映射与 cache 布局知识，实验不复制 hook 实现。

## 公共运行层只管可靠执行，不统一实验科学协议

见[运行层用法](runtime.md)。实验私有 schema、比较条件、结果含义与图表随真实实验加入。
共享组件只在语义稳定且确实需要复用时抽取；不为 E01–E04 建统一的评分或实验基类。

自动记录代码/依赖、配置、seed、状态、实际耗时。模型版本、数据与场景族划分、端口/donor、
forward/backward 次数和显存等由真实组件补充。研究解释链接 run ID，不手抄第二份指标表。

## MLflow 保持可选

```sh
uv sync --locked --extra tracking
uv run --locked --extra tracking python -c "import mlflow; print(mlflow.__version__)"
```

客户端使用 [mlflow-skinny](https://pypi.org/project/mlflow-skinny/)，不在 GPU 机器部署服务端。
`.env.example` 仅说明变量，不自动加载；凭据通过批准机制注入。
本地结果先完整落盘，不启用 autolog，不默认上传模型、数据和全量激活。
新配置显式设置 `artifacts: "off"`，参数/指标与文件同步分别开关。仅直连 OSS/S3 时额外安装：

```sh
uv sync --locked --extra tracking --extra oss
```

`oss` 提供 boto3；归档关闭或使用代理时不需要它，也不检查客户端 OSS 凭据。
代理模式要求对应服务端/实验已经使用代理产物地址，客户端开关不会修改服务端配置。
已提供显式同步和失败重试，见[运行层用法](runtime.md)。可选 SDK 的离线闭环检查：

```sh
uv run --locked --extra tracking pytest tests/runtime/test_tracking_sdk.py
```

该检查只使用临时本地文件后端，创建/记录/上传/读回均已覆盖。默认 CI 检查不安装 tracking；
独立 job 安装可选依赖检查 SDK 接口。实际服务已验证仅参数/指标同步和读回；
实际 OSS 直连/代理的上传与下载尚未验证，不能用离线 SDK 通过代替文件链路验证。
