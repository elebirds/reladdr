"""验证真实运行上下文、状态转换、指标、日志隔离和来源记录。"""

import json
import logging
import random
import subprocess
import sys

import pytest

from reladdr.runtime.config import load_config
from reladdr.runtime.context import RunContext, read_run
from reladdr.runtime.provenance import capture_provenance


def test_local_run_round_trip(runtime_config):
    with RunContext(runtime_config()) as run:
        during = read_run(run.run_dir)
        assert during["status"]["status"] == "running"
        run.record_metadata("dataset", {"revision": "test-1", "scene_families": ["scene-1"]})
        with run.time_phase("write"):
            run.artifacts.write_jsonl(
                "private/rows.jsonl", ({"id": i} for i in range(3)), description="上下文测试数据"
            )
        run.log_metric("measured_count", 3, unit="items")
        run.logger.info("已写入三条测试记录")
    record = read_run(run.run_dir)
    assert record["status"]["status"] == "completed"
    assert record["status"]["wall_time_s"] >= 0
    assert record["metadata"]["validation_scope"] == "experiment"
    assert record["metadata"]["details"]["dataset"]["revision"] == "test-1"
    assert record["tracking"]["status"] == "disabled"
    assert any(
        metric["name"] == "measured_count" and metric["value"] == 3 for metric in record["metrics"]
    )
    assert "已写入三条" in (run.run_dir / "run.log").read_text()
    assert all("accuracy" not in metric["name"] for metric in record["metrics"])


@pytest.mark.parametrize(
    "error, status",
    [
        (RuntimeError("expected failure"), "failed"),
        (KeyboardInterrupt(), "interrupted"),
        (SystemExit(0), "interrupted"),
    ],
)
def test_failure_and_interrupt_preserve_original_error(runtime_config, error, status):
    with pytest.raises(type(error)) as caught, RunContext(runtime_config()) as run:
        run.artifacts.write_json("before-error.json", {"saved": True}, description="异常前记录")
        raise error
    assert caught.value is error
    assert read_run(run.run_dir)["status"]["status"] == status
    assert (run.run_dir / "artifacts/before-error.json").is_file()
    with pytest.raises(RuntimeError):
        run.log_metric("after", 1)
    with pytest.raises(RuntimeError):
        run.artifacts.write_text("after.txt", "invalid", description="结束后不可写")


def test_collision_never_overwrites(runtime_config):
    config = runtime_config()
    with RunContext(config, run_id="fixed") as run:
        pass
    before = (run.run_dir / "status.json").read_bytes()
    with pytest.raises(FileExistsError), RunContext(config, run_id="fixed"):
        pytest.fail("不能进入已存在的 run")
    assert (run.run_dir / "status.json").read_bytes() == before


def test_reentry_is_rejected(runtime_config):
    run = RunContext(runtime_config())
    with run:
        pass
    with pytest.raises(RuntimeError), run:
        pytest.fail("同一上下文不可再次执行")


@pytest.mark.parametrize("run_id", ["../other", "/absolute", ".", "..", "bad/name"])
def test_run_id_cannot_escape(runtime_config, run_id):
    with pytest.raises(ValueError):
        RunContext(runtime_config(), run_id=run_id)


def test_common_only_config_cannot_execute(runtime_config):
    config = runtime_config()
    with pytest.raises(ValueError, match="参数模型"):
        RunContext(load_config(config.source))


def test_rng_and_logger_do_not_modify_global_state(runtime_config):
    before_rng = random.getstate()
    handlers = logging.getLogger().handlers.copy()
    with RunContext(runtime_config()) as first:
        values = [first.rng.random() for _ in range(4)]
    with RunContext(runtime_config()) as second:
        assert values == [second.rng.random() for _ in range(4)]
    assert random.getstate() == before_rng
    assert logging.getLogger().handlers == handlers
    assert first._logger.handlers == []


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "3"])
def test_metric_requires_finite_numeric_value(runtime_config, value):
    with RunContext(runtime_config()) as run, pytest.raises(ValueError):
        run.log_metric("bad", value)


def test_metadata_is_write_once_and_rejects_secrets(runtime_config):
    with RunContext(runtime_config()) as run:
        run.record_metadata("model", {"revision": "fixed"})
        with pytest.raises(ValueError):
            run.record_metadata("model", {"revision": "changed"})
        with pytest.raises(ValueError):
            run.record_metadata("auth", {"password": "never-save"})
    assert "never-save" not in (run.run_dir / "metadata.json").read_text()


def test_auth_values_are_redacted_from_logs_and_exception(runtime_config, monkeypatch):
    secret = "TEST-CREDENTIAL-7392"
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", secret)
    monkeypatch.setenv("UNRELATED_PRIVATE_VARIABLE", "DO-NOT-CAPTURE-THIS")
    with pytest.raises(RuntimeError), RunContext(runtime_config()) as run:
        run.logger.warning("消息 %s", secret)
        run.record_metadata("notes", {"message": secret})
        run.artifacts.write_json("note.json", {"text": secret}, description="测试认证值遮盖")
        raise RuntimeError(secret)
    for path in run.run_dir.rglob("*"):
        if path.is_file():
            text = path.read_text()
            assert secret not in text
            assert "DO-NOT-CAPTURE-THIS" not in text
    assert "[REDACTED]" in (run.run_dir / "run.log").read_text()


def test_initialization_failure_is_recorded(runtime_config, monkeypatch):
    def fail(*args):
        raise RuntimeError("provenance failure")

    monkeypatch.setattr("reladdr.runtime.context.capture_provenance", fail)
    run = RunContext(runtime_config())
    with pytest.raises(RuntimeError, match="provenance failure"), run:
        pytest.fail("初始化失败不可执行主体")
    assert read_run(run.run_dir)["status"]["status"] == "failed"
    assert run._logger.handlers == []


def test_failed_finalization_does_not_mask_body_error(runtime_config, monkeypatch):
    run = RunContext(runtime_config())
    with pytest.raises(RuntimeError, match="body error") as caught, run:

        def fail(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(run, "log_metric", fail)
        raise RuntimeError("body error")
    assert "未完整收尾" in caught.value.__notes__[0]
    assert json.loads((run.run_dir / "status.json").read_text())["status"] == "running"
    assert run._logger.handlers == []


def test_read_run_refuses_external_symlink(runtime_config, tmp_path):
    with RunContext(runtime_config()) as run:
        pass
    outside = tmp_path / "outside.json"
    outside.write_text('{"secret": "do-not-read"}')
    (run.run_dir / "metadata.json").unlink()
    (run.run_dir / "metadata.json").symlink_to(outside)
    with pytest.raises(ValueError, match="越出"):
        read_run(run.run_dir)


def test_git_provenance_captures_tracked_and_new_source(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repository), *args], check=True, capture_output=True, text=True
        )

    git("init")
    (repository / "src").mkdir()
    source = repository / "src/module.py"
    source.write_text("# 测试来源记录。\nvalue = 1\n")
    git("add", "src")
    git(
        "-c",
        "user.name=Runtime Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "test fixture",
    )
    source.write_text("# 测试来源记录。\nvalue = 2\n")
    (repository / "src/new.py").write_text("# 测试新源码记录。\n")
    metadata, diff, untracked = capture_provenance(repository)
    assert metadata["git"]["available"]
    assert metadata["git"]["dirty"]
    assert "+value = 2" in diff
    assert "src/new.py" in untracked
    assert "测试新源码" in untracked["src/new.py"]


@pytest.mark.parametrize("step", [-1, True, 0.5])
def test_metric_step_is_explicit_integer(runtime_config, step):
    with RunContext(runtime_config()) as run, pytest.raises(ValueError):
        run.log_metric("bad_step", 1, step=step)


def test_runtime_import_does_not_load_gpu_or_tracking_stack():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import reladdr.runtime; "
            "assert not {'torch', 'transformers', 'mlflow'} & sys.modules.keys()",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
