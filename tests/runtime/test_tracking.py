"""用受控客户端验证同步白名单、断点、远程身份和失败隔离，不访问网络。"""

import fcntl
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from reladdr.runtime._io import atomic_json
from reladdr.runtime.context import RunContext, read_run
from reladdr.runtime.tracking import TrackingError, sync_run


class ControlledClient:
    """实现同步实际使用的 SDK 接口，并允许在指定调用处注入故障。"""

    def __init__(self, artifact_uri="s3://test-bucket/runs/artifacts"):
        self.artifact_uri = artifact_uri
        self.experiment = None
        self.runs = {}
        self.params = []
        self.metrics = []
        self.uploads = []
        self.finished = []
        self.fail_metric_once = False
        self.lose_create_response_once = False

    def get_experiment_by_name(self, name):
        return self.experiment

    def create_experiment(self, name):
        self.experiment = SimpleNamespace(experiment_id="exp-1")
        return "exp-1"

    def search_runs(self, experiment_ids, filter_string, max_results):
        return [
            run for run in self.runs.values() if run.data.tags["reladdr.run_uid"] in filter_string
        ]

    def create_run(self, experiment_id, start_time, tags, run_name):
        run = SimpleNamespace(
            info=SimpleNamespace(
                run_id=f"remote-{len(self.runs) + 1}",
                experiment_id=experiment_id,
                artifact_uri=self.artifact_uri,
            ),
            data=SimpleNamespace(tags=tags),
        )
        self.runs[run.info.run_id] = run
        if self.lose_create_response_once:
            self.lose_create_response_once = False
            raise ConnectionError("response lost after creation")
        return run

    def get_run(self, run_id):
        return self.runs[run_id]

    def log_param(self, run_id, key, value, *, synchronous):
        assert synchronous
        self.params.append((run_id, key, value))

    def log_metric(self, run_id, key, value, *, timestamp, step, synchronous):
        assert synchronous
        if self.fail_metric_once and self.metrics:
            self.fail_metric_once = False
            raise ConnectionError("controlled metric failure")
        self.metrics.append((run_id, key, value, timestamp, step))

    def log_artifact(self, run_id, local_path, artifact_path):
        self.uploads.append((run_id, Path(local_path).name, artifact_path))

    def set_terminated(self, run_id, *, status, end_time):
        self.finished.append((run_id, status))


@pytest.fixture
def tracked_run(runtime_config, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.invalid")
    with RunContext(runtime_config(tracking="mlflow")) as run:
        run.log_metric("measured_count", 3, unit="items")
        run.log_metric("measured_fraction", 0.5)
        run.artifacts.write_json(
            "summary.json", {"count": 3}, description="同步测试摘要", sync=True
        )
        run.artifacts.write_text("private.txt", "not for upload", description="仅保存在本地")
    return run.run_dir


def test_sync_only_selected_artifacts_and_keep_execution_status(tracked_run):
    client = ControlledClient()
    before = (tracked_run / "status.json").read_bytes()
    state = sync_run(tracked_run, client=client)
    assert state["status"] == "synced"
    assert state["metrics_synced"] == len(read_run(tracked_run)["metrics"])
    assert len(client.runs) == 1
    uploaded = {item[1] for item in client.uploads}
    assert "summary.json" in uploaded
    assert "config.resolved.json" in uploaded
    assert "private.txt" not in uploaded
    assert "source.diff" not in uploaded
    assert client.finished == [("remote-1", "FINISHED")]
    assert (tracked_run / "status.json").read_bytes() == before
    counts = (len(client.metrics), len(client.uploads), len(client.params))
    assert sync_run(tracked_run, client=client)["status"] == "synced"
    assert counts == (len(client.metrics), len(client.uploads), len(client.params))


def test_metric_failure_retry_reuses_remote_run_and_cursor(tracked_run):
    client = ControlledClient()
    client.fail_metric_once = True
    with pytest.raises(TrackingError, match="ConnectionError"):
        sync_run(tracked_run, client=client)
    failed = read_run(tracked_run)
    assert failed["status"]["status"] == "completed"
    assert failed["tracking"]["status"] == "failed"
    assert failed["tracking"]["metrics_synced"] == 1
    state = sync_run(tracked_run, client=client)
    assert state["status"] == "synced"
    assert len(client.runs) == 1
    assert len(client.metrics) == len(read_run(tracked_run)["metrics"])


def test_unknown_create_outcome_is_recovered_by_uid(tracked_run):
    client = ControlledClient()
    client.lose_create_response_once = True
    with pytest.raises(TrackingError):
        sync_run(tracked_run, client=client)
    assert len(client.runs) == 1
    assert read_run(tracked_run)["tracking"]["remote_run_id"] is None
    assert sync_run(tracked_run, client=client)["remote_run_id"] == "remote-1"
    assert len(client.runs) == 1


def test_missing_destination_is_explicit_and_does_not_contact_client(tracked_run, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI")
    client = ControlledClient()
    with pytest.raises(TrackingError, match="MLFLOW_TRACKING_URI"):
        sync_run(tracked_run, client=client)
    assert not client.runs
    assert read_run(tracked_run)["tracking"]["status"] == "failed"


@pytest.mark.parametrize(
    "uri",
    [
        "https://user:SECRET@host/",
        "https://host/?token=SECRET",
        "https://host/#SECRET",
        "https://",
        "not-a-uri",
        "file://remote-host/store",
    ],
)
def test_invalid_or_credentialed_uri_is_not_echoed(tracked_run, monkeypatch, uri):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    client = ControlledClient()
    with pytest.raises(TrackingError) as error:
        sync_run(tracked_run, client=client)
    assert "SECRET" not in str(error.value)
    assert "SECRET" not in (tracked_run / "tracking.json").read_text()
    assert not client.runs


def test_cannot_silently_switch_target(tracked_run, monkeypatch):
    client = ControlledClient()
    sync_run(tracked_run, client=client)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://different.example.invalid")
    with pytest.raises(TrackingError, match="其他同步目标"):
        sync_run(tracked_run, client=client)
    assert len(client.runs) == 1


def test_sync_lock_refuses_concurrent_owner(tracked_run):
    with (tracked_run / ".tracking.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(TrackingError, match="同步进程"):
            sync_run(tracked_run, client=ControlledClient())
    assert read_run(tracked_run)["tracking"]["status"] == "pending"


def test_running_and_local_only_runs_cannot_upload(runtime_config):
    client = ControlledClient()
    with RunContext(runtime_config(tracking="mlflow")) as run:
        with pytest.raises(TrackingError, match="已结束"):
            sync_run(run.run_dir, client=client)
    with RunContext(runtime_config()) as local:
        pass
    with pytest.raises(TrackingError, match="未显式启用"):
        sync_run(local.run_dir, client=client)
    assert not client.runs


def test_registered_artifact_changes_are_rejected_before_remote_calls(tracked_run):
    (tracked_run / "artifacts/summary.json").write_text('{"changed": true}')
    client = ControlledClient()
    with pytest.raises(TrackingError, match="已改变"):
        sync_run(tracked_run, client=client)
    assert not client.runs


def test_oversized_files_are_rejected_before_remote_calls(tracked_run):
    client = ControlledClient()
    with pytest.raises(TrackingError, match="单文件限制"):
        sync_run(tracked_run, client=client, max_artifact_bytes=1)
    assert not client.runs


def test_remote_identity_mismatch_is_rejected(tracked_run):
    client = ControlledClient()
    client.fail_metric_once = True
    with pytest.raises(TrackingError):
        sync_run(tracked_run, client=client)
    client.runs["remote-1"].data.tags["reladdr.run_uid"] = "0" * 32
    with pytest.raises(TrackingError, match="身份不匹配"):
        sync_run(tracked_run, client=client)


def test_sync_error_redacts_known_auth_values(tracked_run, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "SYNC-SECRET-594")
    client = ControlledClient()

    def fail(*args, **kwargs):
        raise RuntimeError("request rejected: SYNC-SECRET-594")

    client.log_param = fail
    with pytest.raises(TrackingError):
        sync_run(tracked_run, client=client)
    assert "SYNC-SECRET-594" not in (tracked_run / "tracking.json").read_text()
    assert read_run(tracked_run)["status"]["status"] == "completed"


def test_external_symlink_is_not_uploaded(tracked_run, tmp_path):
    path = tracked_run / "artifacts/summary.json"
    path.unlink()
    external = tmp_path / "private.json"
    external.write_text("do not upload")
    path.symlink_to(external)
    client = ControlledClient()
    with pytest.raises(TrackingError):
        sync_run(tracked_run, client=client)
    assert not client.runs


def test_invalid_uid_cannot_enter_remote_query(tracked_run):
    path = tracked_run / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["run_uid"] = "' OR '1'='1"
    atomic_json(path, metadata)
    client = ControlledClient()
    with pytest.raises(TrackingError, match="run_uid"):
        sync_run(tracked_run, client=client)
    assert not client.runs


@pytest.mark.parametrize(
    "error, remote_status",
    [
        (RuntimeError("expected failure"), "FAILED"),
        (KeyboardInterrupt(), "KILLED"),
    ],
)
def test_failed_and_interrupted_runs_keep_their_remote_status(
    runtime_config,
    monkeypatch,
    error,
    remote_status,
):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.invalid")
    with pytest.raises(type(error)), RunContext(runtime_config(tracking="mlflow")) as run:
        raise error
    client = ControlledClient()
    sync_run(run.run_dir, client=client)
    assert client.finished == [("remote-1", remote_status)]


def test_artifact_failure_resumes_without_reuploading_acknowledged_files(tracked_run):
    client = ControlledClient()
    original = client.log_artifact
    failed = False

    def upload(*args, **kwargs):
        nonlocal failed
        if len(client.uploads) == 2 and not failed:
            failed = True
            raise ConnectionError("controlled artifact failure")
        original(*args, **kwargs)

    client.log_artifact = upload
    with pytest.raises(TrackingError):
        sync_run(tracked_run, client=client)
    assert len(read_run(tracked_run)["tracking"]["artifacts_synced"]) == 2
    state = sync_run(tracked_run, client=client)
    assert state["status"] == "synced"
    assert len(client.uploads) == len(state["artifacts_synced"])
    assert len(client.runs) == 1


def test_missing_optional_sdk_is_actionable(tracked_run, monkeypatch):
    import builtins

    original_import = builtins.__import__

    def without_sdk(name, *args, **kwargs):
        if name == "mlflow.tracking":
            raise ImportError("controlled missing extra")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_sdk)
    with pytest.raises(TrackingError, match="--extra tracking"):
        sync_run(tracked_run)
    assert read_run(tracked_run)["status"]["status"] == "completed"
    assert read_run(tracked_run)["tracking"]["status"] == "failed"


@pytest.mark.parametrize("params_metrics", [True, False])
@pytest.mark.parametrize("artifacts", ["off", "direct", "proxy"])
def test_independent_switch_combinations(runtime_config, monkeypatch, params_metrics, artifacts):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.invalid")
    uri = "mlflow-artifacts:/1/run/artifacts" if artifacts == "proxy" else "s3://test/run"
    client = ControlledClient(artifact_uri=uri)
    config = runtime_config(tracking="mlflow", params_metrics=params_metrics, artifacts=artifacts)
    with RunContext(config) as run:
        run.log_metric("measured", 2)
        run.artifacts.write_json("summary.json", {}, description="组合检查", sync=True)
    state = sync_run(run.run_dir, client=client)
    assert state["policy"] == {"params_metrics": params_metrics, "artifacts": artifacts}
    assert bool(client.params) == params_metrics
    assert bool(client.metrics) == params_metrics
    assert bool(client.uploads) == (artifacts != "off")
    assert state["params_metrics_status"] == ("synced" if params_metrics else "disabled")
    assert state["artifacts_status"] == ("disabled" if artifacts == "off" else "synced")
    assert state["status"] == (
        "disabled" if not params_metrics and artifacts == "off" else "synced"
    )


def test_off_does_not_inspect_files_or_artifact_backend(tracked_run, monkeypatch):
    (tracked_run / "artifacts/summary.json").unlink()

    def forbidden(*args, **kwargs):
        raise AssertionError("关闭归档时不能检查文件或后端")

    monkeypatch.setattr("reladdr.runtime.tracking._upload_files", forbidden)
    monkeypatch.setattr("reladdr.runtime.tracking._check_artifact_route", forbidden)
    client = ControlledClient()
    state = sync_run(tracked_run, client=client, artifacts="off", max_artifact_bytes=0)
    assert state["params_metrics_status"] == "synced"
    assert not client.uploads
    # 覆盖策略持久化，裸重试不会意外重新启用旧配置的归档。
    assert sync_run(tracked_run, client=client)["policy"]["artifacts"] == "off"
    assert len(client.params) == 4


def test_both_disabled_need_neither_service_nor_sdk(tracked_run, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI")

    def forbidden(*args, **kwargs):
        raise AssertionError("两项关闭后不应创建 SDK 客户端")

    monkeypatch.setattr("reladdr.runtime.tracking._make_client", forbidden)
    state = sync_run(tracked_run, params_metrics=False, artifacts="off")
    assert state["status"] == "disabled"
    assert state["remote_run_id"] is None
    assert sync_run(tracked_run)["status"] == "disabled"


@pytest.mark.parametrize(
    "mode, uri",
    [
        ("direct", "mlflow-artifacts:/1/run/artifacts"),
        ("direct", "file:///tmp/store"),
        ("proxy", "s3://test/run"),
        ("proxy", "https://files.example.invalid/arbitrary"),
        ("proxy", "https://files.example.invalid/api/2.0/mlflow-artifacts/artifacts-other"),
        ("proxy", "mlflow-artifacts:invalid-path"),
    ],
)
def test_route_mismatch_keeps_completed_metrics_without_fallback(tracked_run, mode, uri):
    client = ControlledClient(artifact_uri=uri)
    with pytest.raises(TrackingError, match="不匹配"):
        sync_run(tracked_run, client=client, artifacts=mode)
    state = read_run(tracked_run)["tracking"]
    assert state["params_metrics_status"] == "synced"
    assert state["artifacts_status"] == "failed"
    assert not client.uploads
    count = len(client.metrics)
    state = sync_run(tracked_run, client=client, artifacts="off")
    assert state["status"] == "synced"
    assert len(client.metrics) == count
    assert len(client.params) == 4
    assert len(client.runs) == 1


def test_later_archive_uses_same_run_without_relogging_metrics(tracked_run):
    client = ControlledClient()
    state = sync_run(tracked_run, client=client, artifacts="off")
    count = len(client.metrics)
    config_before = (tracked_run / "config.resolved.json").read_bytes()
    state = sync_run(tracked_run, client=client, params_metrics=False, artifacts="direct")
    assert state["remote_run_id"] == "remote-1"
    assert state["params_metrics_status"] == "disabled"
    assert state["artifacts_status"] == "synced"
    assert len(client.metrics) == count
    assert len(client.runs) == 1
    assert client.uploads
    state = sync_run(tracked_run, client=client, params_metrics=True)
    assert state["params_metrics_status"] == "synced"
    assert len(client.params) == 4
    assert len(client.metrics) == count
    assert (tracked_run / "config.resolved.json").read_bytes() == config_before


def test_proxy_accepts_mlflow_http_artifact_endpoint(tracked_run):
    client = ControlledClient(
        artifact_uri="https://mlflow.example.invalid/api/2.0/mlflow-artifacts/artifacts/1/run"
    )
    assert sync_run(tracked_run, client=client, artifacts="proxy")["status"] == "synced"
    assert client.uploads


def test_archive_preflight_failure_does_not_block_explicit_metrics(tracked_run):
    client = ControlledClient()
    with pytest.raises(TrackingError, match="单文件限制"):
        sync_run(tracked_run, client=client, artifacts="direct", max_artifact_bytes=1)
    state = read_run(tracked_run)["tracking"]
    assert state["params_metrics_status"] == "synced"
    assert state["artifacts_status"] == "failed"
    assert len(client.metrics) == len(read_run(tracked_run)["metrics"])
    assert not client.uploads


def test_auth_failure_does_not_continue_to_object_store(tracked_run):
    client = ControlledClient()

    def denied(*args, **kwargs):
        raise PermissionError("controlled denied")

    client.log_param = denied
    with pytest.raises(TrackingError):
        sync_run(tracked_run, client=client, artifacts="direct")
    assert not client.uploads
    assert read_run(tracked_run)["tracking"]["params_metrics_status"] == "failed"


def test_missing_oss_extra_is_actionable_without_losing_metrics(tracked_run):
    client = ControlledClient()

    def missing(*args, **kwargs):
        raise ModuleNotFoundError("controlled missing boto3", name="boto3")

    client.log_artifact = missing
    with pytest.raises(TrackingError, match="--extra oss"):
        sync_run(tracked_run, client=client, artifacts="direct")
    state = read_run(tracked_run)["tracking"]
    assert state["params_metrics_status"] == "synced"
    assert state["artifacts_status"] == "failed"


def test_old_config_and_state_keep_native_sync_behavior(tracked_run):
    config = json.loads((tracked_run / "config.resolved.json").read_text())
    config["tracking"].pop("params_metrics")
    config["tracking"].pop("artifacts")
    atomic_json(tracked_run / "config.resolved.json", config)
    state = read_run(tracked_run)["tracking"]
    for key in ("policy", "params_metrics_status", "artifacts_status", "parameters_synced"):
        state.pop(key)
    atomic_json(tracked_run / "tracking.json", state)
    client = ControlledClient(artifact_uri="file:///tmp/legacy-store")
    state = sync_run(tracked_run, client=client)
    assert state["policy"]["artifacts"] == "native"
    assert state["status"] == "synced"
    assert client.uploads


@pytest.mark.parametrize("kwargs", [{"params_metrics": "false"}, {"artifacts": "auto"}])
def test_invalid_switches_do_not_contact_remote(tracked_run, kwargs):
    client = ControlledClient()
    with pytest.raises(TrackingError):
        sync_run(tracked_run, client=client, **kwargs)
    assert not client.runs


def test_initial_state_separates_components(runtime_config):
    with RunContext(runtime_config(tracking="mlflow", params_metrics=True, artifacts="off")) as run:
        state = read_run(run.run_dir)["tracking"]
        assert state["status"] == "pending"
        assert state["params_metrics_status"] == "pending"
        assert state["artifacts_status"] == "disabled"
    with RunContext(
        runtime_config(tracking="mlflow", params_metrics=False, artifacts="off")
    ) as run:
        state = read_run(run.run_dir)["tracking"]
        assert state["status"] == "disabled"
        assert state["remote_run_id"] is None


def test_archive_failure_cursor_is_preserved_when_toggling_off(tracked_run):
    client = ControlledClient()
    original = client.log_artifact
    failed = False

    def fail_once(*args, **kwargs):
        nonlocal failed
        if len(client.uploads) == 2 and not failed:
            failed = True
            raise ConnectionError("controlled upload failure")
        original(*args, **kwargs)

    client.log_artifact = fail_once
    with pytest.raises(TrackingError):
        sync_run(tracked_run, client=client, artifacts="direct")
    assert read_run(tracked_run)["tracking"]["params_metrics_status"] == "synced"
    acked = read_run(tracked_run)["tracking"]["artifacts_synced"].copy()
    state = sync_run(tracked_run, client=client, params_metrics=False, artifacts="off")
    assert state["artifacts_synced"] == acked
    state = sync_run(tracked_run, client=client, artifacts="direct")
    assert state["status"] == "synced"
    assert len(client.uploads) == len(state["artifacts_synced"])
    assert len(client.runs) == 1


def test_same_size_same_timestamp_rewrite_is_rejected(tracked_run):
    path = tracked_run / "artifacts/summary.json"
    before = path.stat()
    original = path.read_bytes()
    path.write_bytes(b"x" * len(original))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    client = ControlledClient()
    with pytest.raises(TrackingError, match="内容已改变"):
        sync_run(tracked_run, client=client)
    assert not client.runs
