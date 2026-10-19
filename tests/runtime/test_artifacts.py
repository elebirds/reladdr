"""验证实验产物的隔离、流式记录、登记与禁止覆盖行为。"""

import json

import pytest

from reladdr.runtime._io import atomic_json
from reladdr.runtime.artifacts import ArtifactStore
from reladdr.runtime.context import RunContext, read_run


def test_private_files_and_sync_allowlist(runtime_config):
    with RunContext(runtime_config()) as run:
        store = run.artifacts
        store.write_json("summary.json", {"count": 2}, description="可同步摘要", sync=True)
        store.write_text("raw.txt", "large private data", description="仅本地产物")
        path = store.path_for("plots/result.svg")
        path.write_text("<svg/>")
        store.register("plots/result.svg", description="外部绘图库产物", sync=True)
    entries = read_run(run.run_dir)["manifest"]["artifacts"]
    selected = {entry["path"] for entry in entries if entry["sync"]}
    assert selected == {"artifacts/summary.json", "artifacts/plots/result.svg"}


@pytest.mark.parametrize("name", ["../escape", "/absolute", "nested/../../escape", "."])
def test_path_traversal_is_rejected(runtime_config, name):
    with RunContext(runtime_config()) as run, pytest.raises(ValueError):
        run.artifacts.write_text(name, "bad", description="非法路径")


def test_external_symlink_is_rejected(runtime_config, tmp_path):
    with RunContext(runtime_config()) as run:
        (run.artifacts.directory / "outside").symlink_to(tmp_path, target_is_directory=True)
        with pytest.raises(ValueError):
            run.artifacts.write_text("outside/escape.txt", "bad", description="符号链接逃逸")
    assert not (tmp_path / "escape.txt").exists()


def test_no_overwrite_and_no_duplicate_registration(runtime_config):
    with RunContext(runtime_config()) as run:
        path = run.artifacts.write_text("one.txt", "original", description="原产物")
        with pytest.raises(FileExistsError):
            run.artifacts.write_text("one.txt", "replacement", description="不可覆盖")
        with pytest.raises(ValueError, match="已登记"):
            run.artifacts.register("one.txt", description="重复登记")
        assert path.read_text() == "original"


def test_streaming_failure_is_not_registered(runtime_config):
    def records():
        yield {"sample": 1}
        raise RuntimeError("failed iterator")

    with pytest.raises(RuntimeError), RunContext(runtime_config()) as run:
        run.artifacts.write_jsonl("partial.jsonl", records(), description="未完成逐样本记录")
    partial = run.run_dir / "artifacts/partial.jsonl"
    assert json.loads(partial.read_text()) == {"sample": 1}
    assert not any(
        entry["path"] == "artifacts/partial.jsonl"
        for entry in read_run(run.run_dir)["manifest"]["artifacts"]
    )


def test_atomic_json_keeps_old_file_on_invalid_value(tmp_path):
    path = tmp_path / "state.json"
    atomic_json(path, {"previous": True})
    with pytest.raises(ValueError):
        atomic_json(path, {"invalid": float("nan")})
    assert json.loads(path.read_text()) == {"previous": True}
    assert list(tmp_path.iterdir()) == [path]


def test_closed_store_rejects_mutation(tmp_path):
    store = ArtifactStore(tmp_path)
    store.close()
    with pytest.raises(RuntimeError):
        store.path_for("after.txt")
