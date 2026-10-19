"""通过独立 Python 进程检查真实 CLI、退出码与只读行为。"""

import json
import subprocess
import sys

from reladdr.runtime.context import RunContext


def command(*arguments):
    return subprocess.run(
        [sys.executable, "-m", "reladdr", *map(str, arguments)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_common_check_reports_its_scope_without_starting_run(runtime_config):
    config = runtime_config()
    result = command("config", "check", "--common-only", config.source)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["validation_scope"] == "common_only"
    assert "尚未校验" in report["message"]
    assert not config.document.run.output_root.exists()


def test_common_check_requires_explicit_scope(runtime_config):
    result = command("config", "check", runtime_config().source)
    assert result.returncode == 2
    assert "--common-only" in result.stderr


def test_bad_config_does_not_echo_input(tmp_path):
    path = tmp_path / "secret.yaml"
    path.write_text("password: [DO-NOT-ECHO-123")
    result = command("config", "check", "--common-only", path)
    assert result.returncode == 1
    assert "DO-NOT-ECHO-123" not in result.stdout + result.stderr
    assert "Traceback" not in result.stderr


def test_inspect_json_and_text_are_read_only(runtime_config):
    with RunContext(runtime_config()) as run:
        run.log_metric("measured", 2)
    state_before = (run.run_dir / "status.json").read_bytes()
    structured = command("run", "inspect", "--json", run.run_dir)
    assert structured.returncode == 0, structured.stderr
    assert json.loads(structured.stdout)["status"]["status"] == "completed"
    readable = command("run", "inspect", run.run_dir)
    assert readable.returncode == 0
    assert "completed" in readable.stdout
    assert (run.run_dir / "status.json").read_bytes() == state_before


def test_inspect_does_not_finalize_active_run(runtime_config):
    with RunContext(runtime_config()) as run:
        result = command("run", "inspect", run.run_dir)
        assert result.returncode == 0
        assert "running" in result.stdout
        assert "未完整收尾" in result.stdout


def test_missing_run_returns_clean_failure(tmp_path):
    result = command("run", "inspect", tmp_path / "missing")
    assert result.returncode == 1
    assert "Traceback" not in result.stderr


def test_help_has_no_unimplemented_experiment_command():
    result = command("--help")
    assert result.returncode == 0
    assert "config" in result.stdout
    assert "experiment" not in result.stdout


def test_tracking_cli_requires_explicit_run_opt_in(runtime_config):
    with RunContext(runtime_config()) as run:
        pass
    result = command("tracking", "sync", run.run_dir)
    assert result.returncode == 1
    assert "未显式启用" in result.stderr
    assert "Traceback" not in result.stderr


def test_tracking_cli_can_disable_both_without_sdk_or_destination(runtime_config, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with RunContext(runtime_config(tracking="mlflow")) as run:
        pass
    result = command("tracking", "sync", run.run_dir, "--no-params-metrics", "--artifacts", "off")
    assert result.returncode == 0, result.stderr
    state = json.loads(result.stdout)
    assert state["status"] == "disabled"
    assert state["policy"] == {"params_metrics": False, "artifacts": "off"}
    retry = command("tracking", "sync", run.run_dir)
    assert retry.returncode == 0, retry.stderr
    assert json.loads(retry.stdout)["status"] == "disabled"
    inspected = command("run", "inspect", run.run_dir)
    assert "参数/指标: disabled" in inspected.stdout
    assert "文件归档: disabled (off)" in inspected.stdout


def test_tracking_cli_rejects_unknown_archive_mode(tmp_path):
    result = command("tracking", "sync", tmp_path, "--artifacts", "auto")
    assert result.returncode == 2


def test_tracking_cli_keeps_sdk_console_messages_out_of_json(monkeypatch, capsys, tmp_path):
    from reladdr.cli import main

    def sdk_sync(directory, **kwargs):
        print("SDK run link")
        return {"status": "synced", "policy": kwargs}

    monkeypatch.setattr("reladdr.runtime.tracking.sync_run", sdk_sync)
    assert main(["tracking", "sync", str(tmp_path), "--artifacts", "off"]) == 0
    output = capsys.readouterr()
    assert json.loads(output.out)["status"] == "synced"
    assert "SDK run link" in output.err
