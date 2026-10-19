# RelAddr

**Relation-Conditioned Visual Addressing · 视觉语言模型的关系条件寻址**

研究视觉语言模型如何把参照对象与关系条件接入目标选址，再读取目标属性。
机制研究与联合干预集合搜索是两条独立研究线，共享模型适配和评估基础设施。

当前仓库提供 uv 开发环境和空的 Python 包结构；尚未实现数据生成、模型调用、
激活干预或搜索算法，也没有模型实验结果。

## 开始开发

在仓库根目录运行：

```sh
uv sync --locked
uv run --locked python -c "import reladdr; print(reladdr.__file__)"
```

项目固定 Python 3.12.14，依赖由 `pyproject.toml` 声明、`uv.lock` 锁定。
`uv sync` 使用项目自己的 `.venv`，不要通过全局 pip 安装项目依赖。

如需准备远程 MLflow 客户端：

```sh
uv sync --locked --extra tracking
uv run --locked --extra tracking python -c "import mlflow; print(mlflow.__version__)"
```

这两条命令只安装和导入客户端，不连接服务、不上传数据。
`.env.example` 是变量说明，不会被自动加载。GPU 依赖在模型适配阶段单独确定。

## 包结构

- `reladdr.tasks`：场景、渲染、问题和数据划分。
- `reladdr.models`：模型适配、视觉 token 映射、原生 hook。
- `reladdr.interventions`：缓存、替换和状态恢复。
- `reladdr.mechanisms`：地址表示、关系变换与计算路径分析。
- `reladdr.evaluation`：行为指标、中间响应和成本统计。
- `reladdr.search`：联合干预集合搜索。

以上模块目前只有空的包标记。尚未提供实验 CLI。

## 开发文档

- [开发与复现](docs/development.md)：环境、依赖、GPU/MLflow 接入与运行记录约定。
- [配置规范](configs/README.md)：每个配置应声明的比较条件。
- [测试约定](tests/README.md)：实现时需要验证的研究语义。
- [贡献约定](AGENTS.md)：代码边界、数据与执行权限。

研究计划与推导在独立研究档案中维护，不与原始 run 数据混在一起。
