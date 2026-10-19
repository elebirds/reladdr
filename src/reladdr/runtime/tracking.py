"""显式同步已结束的本地 run；同步失败不改变实验执行状态。"""

import fcntl
import math
import os
import re
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from reladdr.runtime._io import atomic_json, run_path
from reladdr.runtime.context import read_run
from reladdr.runtime.security import redact_text

MAX_ARTIFACT_BYTES = 10 * 1024 * 1024
_TERMINAL_STATUS = {"completed": "FINISHED", "failed": "FAILED", "interrupted": "KILLED"}
_CORE_ARTIFACTS = (
    "config.resolved.json",
    "metadata.json",
    "status.json",
    "metrics.jsonl",
    "manifest.json",
)


class TrackingError(RuntimeError):
    """同步失败或前置条件不成立；本地研究结果仍可读取。"""


@contextmanager
def _sync_lock(root: Path):
    # Linux/macOS 的进程锁随文件关闭自动释放，不留下需要手工删除的“占用”状态。
    with run_path(root, ".tracking.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise TrackingError("该 run 已有同步进程，请等待它完成") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _destination() -> str:
    uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if not uri:
        raise TrackingError("请通过批准的环境变量设置 MLFLOW_TRACKING_URI")
    try:
        parsed = urlsplit(uri)
        valid = parsed.scheme in {"http", "https", "file"}
        if parsed.scheme in {"http", "https"}:
            valid = valid and bool(parsed.hostname)
        elif parsed.scheme == "file":
            valid = valid and parsed.netloc in {"", "localhost"} and parsed.path.startswith("/")
        if not valid or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError
    except ValueError:
        raise TrackingError(
            "服务地址须为 HTTP(S) 或本地 file URI，不能内嵌认证、查询或片段"
        ) from None
    return uri


def _make_client(uri: str):
    try:
        from mlflow.tracking import MlflowClient
    except ImportError:
        raise TrackingError("缺少可选客户端，请使用 uv sync --locked --extra tracking") from None
    return MlflowClient(tracking_uri=uri)


def _upload_files(root: Path, record: dict, limit: int) -> list[tuple[str, Path, str]]:
    uploads = []
    for relative in _CORE_ARTIFACTS:
        uploads.append((relative, run_path(root, relative), "runtime"))
    seen = set(_CORE_ARTIFACTS)
    for entry in record["manifest"]["artifacts"]:
        if not isinstance(entry.get("sync"), bool):
            raise TrackingError("产物同步标记必须是布尔值")
        if not entry["sync"]:
            continue
        relative = entry["path"]
        if (
            not isinstance(relative, str)
            or not relative.startswith("artifacts/")
            or relative in seen
        ):
            raise TrackingError("产物清单包含非法路径或重复文件")
        path = run_path(root, relative)
        stat = path.stat()
        if stat.st_size != entry["size_bytes"] or stat.st_mtime_ns != entry["mtime_ns"]:
            raise TrackingError("登记后的产物已改变，请保留原始 run 并创建新的结果记录")
        seen.add(relative)
        uploads.append((relative, path, Path(relative).parent.as_posix()))
    # 任何本地条件不成立时都先拒绝，避免远程创建后才发现不允许上传。
    for _, path, _ in uploads:
        if not path.is_file() or path.stat().st_size > limit:
            raise TrackingError("同步文件缺失或超过单文件限制，请另行保存大数据并输出小型摘要")
    return uploads


def _check_metrics(metrics: list[dict]) -> None:
    for metric in metrics:
        if not isinstance(metric, dict) or not isinstance(metric.get("name"), str):
            raise TrackingError("指标记录格式无效")
        value = metric.get("value")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise TrackingError("指标记录必须包含有限数值")
        for field in ("step", "timestamp_ms"):
            value = metric.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TrackingError("指标的 step 和 timestamp_ms 必须是非负整数")


def _remote_run(client: Any, record: dict, state: dict, save) -> str:
    uid = record["metadata"]["run_uid"]
    if not isinstance(uid, str) or not re.fullmatch(r"[0-9a-f]{32}", uid):
        raise TrackingError("本地 run_uid 无效")
    name = record["config"]["tracking"]["experiment_name"]
    experiment = client.get_experiment_by_name(name)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(name)
    remote_id = state.get("remote_run_id")
    if remote_id:
        remote = client.get_run(remote_id)
        if (
            remote.data.tags.get("reladdr.run_uid") != uid
            or remote.info.experiment_id != experiment_id
        ):
            raise TrackingError("远程 run 身份不匹配，拒绝写入其他实验记录")
    else:
        # 创建请求可能已成功而响应丢失；重试按稳定 UID 找回，不盲目创建第二个 run。
        found = client.search_runs(
            [experiment_id],
            filter_string=f"tags.`reladdr.run_uid` = '{uid}'",
            max_results=2,
        )
        if len(found) > 1:
            raise TrackingError("远程存在多个相同 UID 的 run，请先处理重复记录")
        if found:
            remote_id = found[0].info.run_id
        else:
            start = datetime.fromisoformat(record["status"]["started_at"])
            remote = client.create_run(
                experiment_id,
                start_time=int(start.timestamp() * 1000),
                tags={
                    "reladdr.run_uid": uid,
                    "reladdr.local_run_id": record["metadata"]["run_id"],
                    "reladdr.experiment_id": record["metadata"]["experiment"]["id"],
                },
                run_name=record["metadata"]["run_id"],
            )
            remote_id = remote.info.run_id
        state["remote_run_id"] = remote_id
        save()
    return remote_id


def sync_run(
    directory: Path | str, *, client: Any = None, max_artifact_bytes: int = MAX_ARTIFACT_BYTES
) -> dict:
    """同步终态 run；client 参数用于受控测试，正常调用使用可选 MLflow SDK。"""
    root = Path(directory).expanduser().resolve()
    try:
        record = read_run(root)
    except (OSError, ValueError, KeyError, TypeError):
        raise TrackingError("无法读取完整的本地 run") from None
    if record["status"].get("status") not in _TERMINAL_STATUS:
        raise TrackingError("仅同步已结束的 run；不会替运行中或未收尾的记录补写成功状态")
    if record["config"]["tracking"]["mode"] != "mlflow":
        raise TrackingError("此 run 未显式启用 MLflow，本地记录不会自动上传")
    if (
        isinstance(max_artifact_bytes, bool)
        or not isinstance(max_artifact_bytes, int)
        or max_artifact_bytes <= 0
    ):
        raise TrackingError("单文件同步限制必须为正整数")
    with _sync_lock(root):
        record = read_run(root)
        state = record["tracking"].copy()

        def save():
            atomic_json(root / "tracking.json", state)

        try:
            uri = _destination()
            if state.get("destination") and state["destination"] != redact_text(uri):
                raise TrackingError("该 run 已绑定其他同步目标，拒绝静默改投服务")
            uploads = _upload_files(root, record, max_artifact_bytes)
            _check_metrics(record["metrics"])
            cursor = state.get("metrics_synced", 0)
            if (
                isinstance(cursor, bool)
                or not isinstance(cursor, int)
                or not 0 <= cursor <= len(record["metrics"])
            ):
                raise TrackingError("同步进度与本地指标不一致")
            if state["status"] == "synced":
                return state
            state.update({"status": "syncing", "destination": uri})
            state.pop("error", None)
            save()
            active_client = client if client is not None else _make_client(uri)
            remote_id = _remote_run(active_client, record, state, save)
            identity = record["config"]["experiment"]
            parameters = {
                "experiment.id": identity["id"],
                "experiment.protocol_version": identity["protocol_version"],
                "run.seed": record["config"]["run"]["seed"],
                "config.schema_version": record["config"]["schema_version"],
            }
            # 完整私有参数在配置产物中保存，不猜测 MLflow 扁平参数的科学含义。
            for name, value in parameters.items():
                active_client.log_param(remote_id, name, value, synchronous=True)
            for index, metric in enumerate(record["metrics"][cursor:], start=cursor):
                active_client.log_metric(
                    remote_id,
                    metric["name"],
                    metric["value"],
                    timestamp=metric["timestamp_ms"],
                    step=metric["step"],
                    synchronous=True,
                )
                state["metrics_synced"] = index + 1
                save()
            for relative, path, remote_directory in uploads:
                if relative not in state["artifacts_synced"]:
                    active_client.log_artifact(remote_id, str(path), artifact_path=remote_directory)
                    state["artifacts_synced"].append(relative)
                    save()
            end = datetime.fromisoformat(record["status"]["ended_at"])
            active_client.set_terminated(
                remote_id,
                status=_TERMINAL_STATUS[record["status"]["status"]],
                end_time=int(end.timestamp() * 1000),
            )
            state["status"] = "synced"
            save()
            return state
        except Exception as exc:
            state["status"] = "failed"
            state["error"] = {"type": type(exc).__name__, "message": redact_text(str(exc))}
            try:
                save()
            except OSError as write_error:
                raise TrackingError(
                    "同步失败且无法更新 tracking.json，请检查本地磁盘"
                ) from write_error
            if isinstance(exc, TrackingError):
                raise
            raise TrackingError(
                f"MLflow 同步失败（{type(exc).__name__}），详见 tracking.json"
            ) from None
