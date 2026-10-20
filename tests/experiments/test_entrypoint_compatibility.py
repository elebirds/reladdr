"""验证新子包与旧入口共享实现、保留配置身份，并且离线 CLI 不加载模型。"""

import importlib
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from reladdr.runtime import load_config, read_run

EXPERIMENTS = (
    (
        "e01_formal",
        "b01_behavioral_baseline.runner",
        "run_formal",
        "E01",
        "B01_Behavioral_Baseline",
        "e01-v1",
        ("b01_behavioral_baseline",),
        "dataset/scenes.jsonl",
    ),
    (
        "e02",
        "b02_binding_replication.runner",
        "run_e02",
        "E02",
        "B02_Binding_Replication",
        "e02-v1",
        ("b02_binding_replication",),
        "dataset/pairs.jsonl",
    ),
    (
        "e03",
        "m01_interface_diagnostics.linear",
        "run_e03",
        "E03",
        "M01_Interface_Diagnostics",
        "e03a-v2",
        ("m01_interface_diagnostics", "linear"),
        "dataset/pairs.jsonl",
    ),
    (
        "e03_grid",
        "m01_interface_diagnostics.grid",
        "run_grid",
        "E03",
        "M01_Interface_Diagnostics",
        "e03a-grid-v1",
        ("m01_interface_diagnostics", "grid"),
        "dataset/pairs.jsonl",
    ),
    (
        "e03_path",
        "m02_path_diagnostics.mediation",
        "run_path",
        "E03",
        "M02_Path_Diagnostics",
        "e03b-mediation-v1",
        ("m02_path_diagnostics", "mediation"),
        "dataset/pairs.jsonl",
    ),
    (
        "e03_freeze",
        "m02_path_diagnostics.freeze",
        "run_freeze",
        "E03",
        "M02_Path_Diagnostics",
        "e03b-conditional-freeze-v1",
        ("m02_path_diagnostics", "freeze"),
        "dataset/cases.jsonl",
    ),
)


def write_config(root, experiment_id, protocol, *, formal=False):
    root.mkdir(parents=True, exist_ok=True)
    parameters = {
        "model_path": "must-not-load",
        "model_revision": "offline-contract-test",
        "family_counts": {"discovery": 1, "validation": 1, "test": 1},
    }
    if formal:
        parameters.update(layouts=["horizontal"], bootstrap_replicates=100)
    source = root / "config.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "experiment": {
                    "id": experiment_id,
                    "protocol_version": protocol,
                    "parameters": parameters,
                },
                "run": {"seed": 19, "output_root": "output"},
                "tracking": {"mode": "local", "artifacts": "off"},
            }
        )
    )
    return source


@pytest.mark.parametrize("case", EXPERIMENTS, ids=lambda case: case[0])
def test_old_import_is_actual_module_and_keeps_public_entrypoint(case):
    legacy, canonical, runner, *_ = case
    old = importlib.import_module(f"reladdr.experiments.{legacy}")
    new = importlib.import_module(f"reladdr.experiments.{canonical}")
    assert old is new
    assert old.Parameters is new.Parameters
    assert getattr(old, runner) is getattr(new, runner)


def test_old_entrypoint_monkeypatch_reaches_real_execution(monkeypatch):
    old = importlib.import_module("reladdr.experiments.e02")
    new = importlib.import_module("reladdr.experiments.b02_binding_replication.runner")
    calls = []
    expected = object()

    def patched(config, pairs, **kwargs):
        calls.append((config, pairs, kwargs))
        return expected

    monkeypatch.setattr(old, "run_patching", patched)
    config = SimpleNamespace(
        document=SimpleNamespace(
            experiment=SimpleNamespace(protocol_version="e02-v1"),
            run=SimpleNamespace(seed=19),
        ),
        parameters=old.Parameters(
            model_path="unused",
            model_revision="offline-test",
            family_counts={"discovery": 1, "validation": 1, "test": 1},
        ),
    )
    assert new.run_e02(config, prepare_only=True) is expected
    assert len(calls) == 1
    assert calls[0][2]["prepare_only"]


@pytest.mark.parametrize("case", EXPERIMENTS, ids=lambda case: case[0])
def test_new_and_old_ids_preserve_identity_and_identical_data(tmp_path, monkeypatch, case):
    legacy, canonical, runner_name, old_id, new_id, protocol, _, artifact = case
    module = importlib.import_module(f"reladdr.experiments.{canonical}")

    def forbidden(*args, **kwargs):
        pytest.fail("prepare-only must not construct a model")

    # 先导入原生适配器类定义，再拦截构造；导入本身不加载 Torch 或模型。
    importlib.import_module("reladdr.models.qwen2_native")
    monkeypatch.setattr("reladdr.models.qwen2_vl.Qwen2VLAdapter", forbidden)
    monkeypatch.setattr("reladdr.models.qwen2_native.Qwen2NativeAdapter", forbidden)
    directories = []
    for experiment_id in (old_id, new_id):
        source = write_config(
            tmp_path / experiment_id, experiment_id, protocol, formal=legacy == "e01_formal"
        )
        config = load_config(source, parameter_model=module.Parameters)
        directory = getattr(module, runner_name)(config, prepare_only=True)
        record = read_run(directory)
        assert record["metadata"]["experiment"] == {
            "id": experiment_id,
            "protocol_version": protocol,
        }
        resolved = json.loads((directory / "config.resolved.json").read_text())
        assert resolved["experiment"]["id"] == experiment_id
        assert resolved["experiment"]["protocol_version"] == protocol
        assert record["status"]["status"] == "completed"
        assert not record["metadata"]["details"]["protocol"]["evaluation_executed"]
        directories.append(directory)
    assert (directories[0] / "artifacts" / artifact).read_text() == (
        directories[1] / "artifacts" / artifact
    ).read_text()
    assert (
        read_run(directories[0])["metadata"]["details"]["protocol"]
        == read_run(directories[1])["metadata"]["details"]["protocol"]
    )


@pytest.mark.parametrize("case", EXPERIMENTS, ids=lambda case: case[0])
@pytest.mark.parametrize("invalid", ["identity", "protocol"])
def test_wrong_experiment_or_protocol_fails_before_creating_run(tmp_path, case, invalid):
    legacy, canonical, runner_name, _, new_id, protocol, *_ = case
    module = importlib.import_module(f"reladdr.experiments.{canonical}")
    source = write_config(
        tmp_path,
        "Unrelated_Experiment" if invalid == "identity" else new_id,
        "unrelated-v1" if invalid == "protocol" else protocol,
        formal=legacy == "e01_formal",
    )
    config = load_config(source, parameter_model=module.Parameters)
    with pytest.raises(ValueError):
        getattr(module, runner_name)(config, prepare_only=True)
    assert not config.document.run.output_root.exists()


@pytest.mark.parametrize(
    "command",
    [
        ("e01",),
        *[(case[0],) for case in EXPERIMENTS],
        ("b01_behavioral_baseline",),
        ("b02_binding_replication",),
        ("m01_interface_diagnostics",),
        ("m02_path_diagnostics",),
        ("m01_interface_diagnostics", "linear"),
        ("m01_interface_diagnostics", "grid"),
        ("m02_path_diagnostics", "mediation"),
        ("m02_path_diagnostics", "freeze"),
    ],
)
def test_real_module_cli_help_is_available_without_gpu(command):
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            f"reladdr.experiments.{command[0]}",
            *command[1:],
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "usage:" in completed.stdout
    assert "RuntimeWarning" not in completed.stderr


@pytest.mark.parametrize("case", EXPERIMENTS, ids=lambda case: case[0])
@pytest.mark.parametrize("legacy_cli", [False, True])
def test_real_prepare_cli_records_declared_identity_without_gpu(tmp_path, case, legacy_cli):
    legacy, _, _, old_id, new_id, protocol, command, _ = case
    experiment_id = old_id if legacy_cli else new_id
    source = write_config(tmp_path, experiment_id, protocol, formal=legacy == "e01_formal")
    command = (legacy,) if legacy_cli else command
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            f"reladdr.experiments.{command[0]}",
            *command[1:],
            str(source),
            "--prepare-only",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    directory = next((tmp_path / "output").iterdir())
    assert read_run(directory)["metadata"]["experiment"]["id"] == experiment_id
    assert not list(directory.glob("artifacts/*/predictions.jsonl"))
