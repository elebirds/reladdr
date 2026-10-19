"""用受控客户端验证同步白名单、断点、远程身份和失败隔离，不访问网络。"""

import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from reladdr.runtime._io import atomic_json
from reladdr.runtime.context import RunContext, read_run
from reladdr.runtime.tracking import TrackingError, sync_run


class ControlledClient:
    """实现同步实际使用的 SDK 接口，并允许在指定调用处注入故障。"""

    def __init__(self):
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
                run_id=f"remote-{len(self.runs) + 1}", experiment_id=experiment_id
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
