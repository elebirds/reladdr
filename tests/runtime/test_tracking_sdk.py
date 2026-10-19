"""用真实 MLflow SDK 和临时文件后端验证同步与读回，全程不访问远程服务。"""

import json
import subprocess
import sys

import pytest

from reladdr.runtime.context import RunContext
from reladdr.runtime.tracking import sync_run

mlflow = pytest.importorskip("mlflow", reason="真实 SDK 检查需要可选 tracking 依赖")


def test_real_sdk_file_store_round_trip(runtime_config, tmp_path, monkeypatch):
    uri = (tmp_path / "mlflow-store").as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    # 文件后端只用于隔离 SDK 验证；不修改用户环境，也不作为实际服务部署方案。
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    with RunContext(runtime_config(tracking="mlflow")) as run:
        run.log_metric("measured_count", 4, unit="items")
        run.artifacts.write_json(
            "summary.json", {"count": 4}, description="真实 SDK 测试", sync=True
        )
        run.artifacts.write_text("private.txt", "do not sync", description="仅本地文件")
    result = subprocess.run(
        [sys.executable, "-m", "reladdr", "tracking", "sync", str(run.run_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    state = json.loads(result.stdout)
    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    remote = client.get_run(state["remote_run_id"])
    assert remote.info.status == "FINISHED"
    assert remote.data.params["experiment.id"] == "runtime_test"
    assert remote.data.metrics["measured_count"] == 4
    files = client.list_artifacts(state["remote_run_id"], "artifacts")
    assert {file.path for file in files} == {"artifacts/summary.json"}
    path = client.download_artifacts(state["remote_run_id"], "artifacts/summary.json")
    with open(path, encoding="utf-8") as stream:
        assert json.load(stream) == {"count": 4}
    assert sync_run(run.run_dir)["remote_run_id"] == remote.info.run_id


def test_real_sdk_cli_metrics_only_has_no_files(runtime_config, tmp_path, monkeypatch):
    uri = (tmp_path / "mlflow-metadata-store").as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    with RunContext(runtime_config(tracking="mlflow", artifacts="off")) as run:
        run.log_metric("measured", 7)
        run.artifacts.write_json("local.json", {}, description="关闭归档的登记产物", sync=True)
    # 文件缺失不能阻塞关闭归档时的参数/指标同步。
    (run.run_dir / "artifacts/local.json").unlink()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "reladdr",
            "tracking",
            "sync",
            str(run.run_dir),
            "--params-metrics",
            "--artifacts",
            "off",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    state = json.loads(result.stdout)
    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    remote = client.get_run(state["remote_run_id"])
    assert remote.info.status == "FINISHED"
    assert remote.data.params["run.seed"] == "19"
    assert remote.data.metrics["measured"] == 7
    assert client.list_artifacts(remote.info.run_id) == []
    assert state["params_metrics_status"] == "synced"
    assert state["artifacts_status"] == "disabled"
    # 裸重试沿用 off 且不重复发送已经确认的指标。
    assert sync_run(run.run_dir)["remote_run_id"] == remote.info.run_id
    assert len(client.get_metric_history(remote.info.run_id, "measured")) == 1
