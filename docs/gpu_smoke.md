<!-- 说明首个真实 GPU 调试入口、输入真值、环境边界与结果解读。 -->
# 用 Instruct 跑通三形状初调

这条入口检查模型加载、预处理、答案读出、图像依赖和运行记录。它生成一个场景族的横/竖
布局：每张图有不同颜色的方形、圆形和三角形，真值来自生成时的对象与位置，不从模型答案反推。

首轮实测见 [2026-10-19 GPU 初调记录](reports/gpu_smoke_2026-10-19.md)。

## 完整环境的运行方式

```sh
uv sync --locked --extra gpu
uv run --locked --extra gpu python -m reladdr.experiments.e01 configs/e01_smoke.local.yaml
```

将 `configs/e01_smoke.yaml` 复制为忽略的 `.local.yaml`，填入本地 **Instruct** 模型路径和
实际 checkpoint 来源。模型只从本地读取，运行入口不会下载模型，也不自动改用 Base、CPU 或量化。
示例的 `unverified-local-checkpoint` 明确表示尚未提供固定版本，不能据此声称模型版本已锁定。

`families` 默认 1：两张布局图、每图 10 问，三种图像条件共 60 次评估。
所有样本标为 `debug`；正式 discovery/validation/test 分组与网格覆盖继续按研究协议实现。

## 三类问题分别检查什么

- **recognition**：按形状识别对象，再回答其颜色，例如圆形是什么颜色。
- **absolute**：读取最左/中间/最右或最上/中间/最下对象的颜色。
- **relation**：先给参照形状，再问其直接左/右/上/下对象的颜色。边界外没有有效目标时不出题。

图像固定为 448×448，默认对应 256 个合并后的视觉 token；以实际处理结果为准。
颜色与形状独立抽样，横/竖变体保留共同 `family_id`。单个调试族不足以支持总体准确率或机制结论。

## 对照不能混成同一种正确率

同一问题分别接收原图、全白图、保持形状/位置而循环置换颜色的新图。保存实际输入的每张图。

- 原图与颜色错配图按各自实际呈现的颜色计算 `accuracy`。
- 空白图没有真实目标颜色，不生成准确率。
- 三种条件都记录 `original_label_match_rate` 和 `mean_original_answer_margin`，观察对原答案的依赖。
- foil 在生成问题时固定，三种条件共用，不按每次输出重新挑选。

自由生成要求单个颜色词；单词格式与是否属于六色词表分开记录。例如空白图回答 white
不属于六个候选，但不算多词格式失败。原始文本始终保留。下一 token 读出合并颜色的小写/首字母
大写别名，保存完整词表概率与候选 token ID；边界上不是单 token 时明确失败，不静默截断。

## 当前 AutoDL 镜像环境

本次初调的服务器工程、模型和数据分别放在：

- 工程：`/root/autodl-tmp/reladdr`
- 模型：`/root/autodl-tmp/models/Qwen2-VL-7B-Instruct`
- 前作素材：`/root/autodl-tmp/reladdr-data/raw/spatial-variable-binding-main/shared`
- 下载与环境记录：`/root/autodl-tmp/reladdr-ops`

镜像已有 Torch 2.8.0+cu128、torchvision 0.23.0+cu128。初调先对照锁文件核验继承包版本，
只在项目 `.venv` 补齐缺项，不修改镜像全局 Python。运行已准备的镜像环境：

```sh
cd /root/autodl-tmp/reladdr
/root/autodl-tmp/reladdr-tools/python/bin/uv run --locked --no-sync \
  --python /root/miniconda3/bin/python python -m reladdr.experiments.e01 configs/e01_smoke.local.yaml
```

此环境使用 Python 3.12.3，而项目默认是 3.12.14；实际版本和分发包清单写入每次 metadata。
`--no-sync` 不会应用后续锁文件变更，改依赖后须重新准备环境。正式复现优先使用上面的完整环境。

本次关闭远程 Tracking 与 OSS。不会自动上传 `.env`、模型、图片或运行产物。

## 输出从哪里看

每次运行创建独立的 `runs/<run-id>/`，保存：

- `artifacts/dataset/`：原图和实际图像条件、场景/对象真值、问题与固定 foil。
- `artifacts/baseline/predictions.jsonl`：原始答案、候选概率、token 数、耗时、格式和标签判定。
- `artifacts/baseline/summary.json`：按图像条件与问题种类切片的汇总。
- `metadata.json`、`config.resolved.json`、`status.json`：实际模型栈、来源、配置与执行状态。

显存峰值包括加载与评估；运行结束后进程释放 GPU，不常驻模型服务。失败保留已有本地记录，
不把工程执行成功当作机制假说得到验证。
