"""管理一次本地执行的生命周期、日志、指标和可追溯结果。"""

import json
import logging
import math
import os
import random
import re
import socket
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from reladdr.runtime._io import atomic_json, clean_json, parse_json, read_json, run_path
from reladdr.runtime.artifacts import ArtifactStore
from reladdr.runtime.config import LoadedConfig
from reladdr.runtime.provenance import capture_provenance
from reladdr.runtime.security import redact_text


def _now() -> str:
    return datetime.now(UTC).isoformat()


class _RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        # 最终格式化后再遮盖，连异常 traceback 中的已知认证值一起处理。
        return redact_text(super().format(record))


class RunContext:
    """一个上下文对应一次执行；不提供恢复执行或自动远程同步。"""

    def __init__(
        self, config: LoadedConfig, *, run_id: str | None = None, code_root: Path | None = None
    ):
        if config.parameters is None:
            raise ValueError("运行前必须通过实验自己的参数模型完成校验")
        generated = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        label = config.document.run.label or config.document.experiment.id
        self.run_id = (
            run_id if run_id is not None else f"{generated}-{label}-{uuid.uuid4().hex[:8]}"
        )
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", self.run_id):
            raise ValueError("run ID 只能包含字母、数字、下划线、短横线和点")
        if self.run_id in {".", ".."}:
            raise ValueError("无效 run ID")
        self.config = config
        self.run_dir = config.document.run.output_root / self.run_id
        self.rng = random.Random(config.document.run.seed)
        self._code_root = code_root
        self._entered = False
        self._active = False
        self._logger: logging.Logger | None = None
        self._handlers: list[logging.Handler] = []
        self.artifacts: ArtifactStore | None = None
        self._metadata: dict = {}

    @property
    def logger(self) -> logging.Logger:
        self._check_active()
        assert self._logger is not None
        return self._logger

    def _check_active(self) -> None:
        if not self._active:
            raise RuntimeError("run 尚未开始或已经结束")

    def __enter__(self):
        if self._entered:
            raise RuntimeError("同一个 RunContext 不能重复执行")
        snapshot = self.config.snapshot()
        self.run_dir.parent.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir()  # 原子独占创建：即使 ID 冲突，也绝不覆盖旧结果。
        self._entered = True
        self._active = True
        self._started_clock = time.perf_counter()
        self._started_at = _now()
        try:
            atomic_json(
                self.run_dir / "status.json",
                {
                    "schema_version": 1,
                    "status": "running",
                    "started_at": self._started_at,
                    "ended_at": None,
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                },
            )
            atomic_json(self.run_dir / "config.resolved.json", snapshot)
            self._metadata = {
                "schema_version": 1,
                "run_id": self.run_id,
                "run_uid": uuid.uuid4().hex,
                "experiment": {
                    "id": snapshot["experiment"]["id"],
                    "protocol_version": snapshot["experiment"]["protocol_version"],
                },
                "config_source": str(self.config.source),
                "validation_scope": self.config.validation_scope,
                "seed": snapshot["run"]["seed"],
                "details": {},
            }
            atomic_json(self.run_dir / "metadata.json", self._metadata)
            tracking = snapshot["tracking"]
            params_enabled = tracking["mode"] == "mlflow" and tracking["params_metrics"]
            artifacts_enabled = tracking["mode"] == "mlflow" and tracking["artifacts"] != "off"
            atomic_json(
                self.run_dir / "tracking.json",
                {
                    "schema_version": 1,
                    "mode": snapshot["tracking"]["mode"],
                    "status": "pending" if params_enabled or artifacts_enabled else "disabled",
                    "policy": {
                        "params_metrics": tracking["params_metrics"],
                        "artifacts": tracking["artifacts"] or "native",
                    },
                    "params_metrics_status": "pending" if params_enabled else "disabled",
                    "artifacts_status": "pending" if artifacts_enabled else "disabled",
                    "parameters_synced": False,
                    "remote_run_id": None,
                    "metrics_synced": 0,
                    "artifacts_synced": [],
                },
            )
            (self.run_dir / "metrics.jsonl").touch(exist_ok=False)
            self.artifacts = ArtifactStore(self.run_dir)
            self._setup_logger()
            provenance, diff, sources = capture_provenance(self._code_root)
            self._metadata["provenance"] = provenance
            atomic_json(self.run_dir / "metadata.json", self._metadata)
            if diff:
                self.artifacts.write_text(
                    "provenance/source.diff", diff, description="相关源码差异"
                )
            if sources:
                self.artifacts.write_json(
                    "provenance/untracked_sources.json", sources, description="未跟踪源码快照"
                )
            self.logger.info("执行开始：%s", self.run_id)
            return self
        except BaseException as exc:
            self._finish(exc)
            raise

    def _setup_logger(self) -> None:
        logger = logging.Logger(f"reladdr.run.{self.run_id}")
        logger.setLevel(self.config.document.run.log_level)
        logger.propagate = False
        formatter = _RedactingFormatter("%(asctime)s %(levelname)s %(message)s")
        file_handler = logging.FileHandler(self.run_dir / "run.log", encoding="utf-8")
        self._handlers = [file_handler, logging.StreamHandler()]
        for handler in self._handlers:
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        self._logger = logger

    def record_metadata(self, section: str, values: dict) -> None:
        """实验自行记录模型/数据/端口等信息；每个节只允许写入一次。"""
        self._check_active()
        if not section.strip() or section in self._metadata["details"]:
            raise ValueError("metadata 节名为空或已存在")
        clean = clean_json({section: values})
        # 先写盘成功，再更新内存，避免出现看似已经记录的状态。
        updated = {**self._metadata, "details": {**self._metadata["details"], **clean}}
        atomic_json(self.run_dir / "metadata.json", updated)
        self._metadata = updated

    def log_metric(
        self, name: str, value: float, *, step: int = 0, unit: str | None = None
    ) -> None:
        self._check_active()
        if not name.strip() or any(character in name for character in "\r\n"):
            raise ValueError("指标名不能为空或包含换行")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("指标必须是有限数值")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("step 必须是非负整数")
        record = {
            "name": name,
            "value": value,
            "step": step,
            "unit": unit,
            "timestamp_ms": time.time_ns() // 1_000_000,
        }
        text = json.dumps(clean_json(record), ensure_ascii=False, allow_nan=False)
        with (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(text + "\n")

    @contextmanager
    def time_phase(self, name: str):
        """无论阶段成功与否都记录实际耗时，不自动生成其他成本数字。"""
        self._check_active()
        start = time.perf_counter()
        body_error = None
        try:
            yield
        except BaseException as exc:
            body_error = exc
            raise
        finally:
            try:
                self.log_metric(f"phase.{name}.wall_time_s", time.perf_counter() - start, unit="s")
            except BaseException as record_error:
                # 记录失败不能掩盖昂贵实验原先抛出的错误。
                if body_error is None:
                    raise
                body_error.add_note(f"阶段耗时未保存：{type(record_error).__name__}")

    def _finish(self, exc: BaseException | None) -> None:
        status = "completed" if exc is None else "failed"
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            status = "interrupted"
        try:
            elapsed = time.perf_counter() - self._started_clock
            if (self.run_dir / "metrics.jsonl").exists():
                self.log_metric("runtime.wall_time_s", elapsed, unit="s")
            if self._logger is not None:
                if exc is not None:
                    self._logger.error(
                        "执行%s", status, exc_info=(type(exc), exc, exc.__traceback__)
                    )
                else:
                    self._logger.info("执行完成，耗时 %.3f s", elapsed)
            state = read_json(self.run_dir / "status.json")
            state.update({"status": status, "ended_at": _now(), "wall_time_s": elapsed})
            if exc is not None:
                state["error"] = {"type": type(exc).__name__, "message": redact_text(str(exc))}
            atomic_json(self.run_dir / "status.json", state)
        except BaseException as finalization_error:
            if exc is None:
                raise
            exc.add_note(f"运行记录未完整收尾：{type(finalization_error).__name__}")
        finally:
            self._active = False
            if self.artifacts is not None:
                self.artifacts.close()
            for handler in self._handlers:
                handler.close()
                if self._logger is not None:
                    self._logger.removeHandler(handler)

    def __exit__(self, exc_type, exc, traceback):
        self._finish(exc)
        return False


def read_run(directory: Path | str) -> dict:
    """读取本地记录；运行中或未收尾的记录保持原状态，不推断它已成功。"""
    root = Path(directory).expanduser().resolve()
    result = {
        "run_dir": str(root),
        **{
            name: read_json(run_path(root, file))
            for name, file in (
                ("config", "config.resolved.json"),
                ("metadata", "metadata.json"),
                ("status", "status.json"),
                ("tracking", "tracking.json"),
                ("manifest", "manifest.json"),
            )
        },
    }
    for name in ("config", "metadata", "status", "tracking", "manifest"):
        if (
            not isinstance(result[name], dict)
            or type(result[name].get("schema_version")) is not int
            or result[name]["schema_version"] != 1
        ):
            raise ValueError("无法读取此版本的运行记录")
    result["metrics"] = []
    with run_path(root, "metrics.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                result["metrics"].append(parse_json(line))
    return clean_json(result)
