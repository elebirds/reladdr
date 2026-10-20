"""验证新版实验入口、严格身份和协议、离线准备及无 GPU 依赖的导入。"""

import importlib
import json
import subprocess
import sys
from dataclasses import dataclass

import pytest
import yaml

from reladdr.runtime import load_config, read_run


@dataclass(frozen=True)
class Case:
    module: str
    runner: str
    experiment_id: str
    protocol: str
    command: tuple[str, ...]
    artifact: str


CASES = (
    Case(
        "b01_behavioral_baseline.runner",
        "run_formal",
        "B01_Behavioral_Baseline",
        "b01-v1",
        ("b01_behavioral_baseline",),
        "dataset/scenes.jsonl",
    ),
    Case(
        "b01_behavioral_baseline.smoke",
        "run_baseline",
        "B01_Behavioral_Baseline",
        "b01-engineering-v1",
        ("b01_behavioral_baseline.smoke",),
        "dataset/scenes.jsonl",
    ),
    *(
        Case(
            "b02_binding_replication.runner",
            "run_replication",
            "B02_Binding_Replication",
            protocol,
            ("b02_binding_replication",),
            "dataset/pairs.jsonl",
        )
        for protocol in ("b02-directional-v1", "b02-adjacent-v1")
    ),
    *(
        Case(
            "m01_interface_diagnostics.linear",
            "run_linear",
            "M01_Interface_Diagnostics",
            protocol,
            ("m01_interface_diagnostics", "linear"),
            "dataset/pairs.jsonl",
        )
        for protocol in ("m01-directional-v1", "m01-adjacent-v1")
    ),
    Case(
        "m01_interface_diagnostics.grid",
        "run_grid",
        "M01_Interface_Diagnostics",
        "m01-grid-v1",
        ("m01_interface_diagnostics", "grid"),
        "dataset/pairs.jsonl",
    ),
    Case(
        "m02_path_diagnostics.mediation",
        "run_path",
        "M02_Path_Diagnostics",
        "m02-suffix-mediation-v1",
        ("m02_path_diagnostics", "mediation"),
        "dataset/pairs.jsonl",
    ),
    Case(
        "m02_path_diagnostics.freeze",
        "run_freeze",
        "M02_Path_Diagnostics",
        "m02-conditional-freeze-v1",
        ("m02_path_diagnostics", "freeze"),
        "dataset/cases.jsonl",
    ),
)
PREPARABLE_CASES = tuple(case for case in CASES if case.runner != "run_baseline")


def write_config(root, case, *, experiment_id=None, protocol=None):
    root.mkdir(parents=True, exist_ok=True)
    parameters = {"model_path": "must-not-load", "model_revision": "offline-contract-test"}
    if case.runner == "run_baseline":
        parameters["families"] = 1
    else:
        parameters["family_counts"] = {"discovery": 1, "validation": 1, "test": 1}
    if case.runner == "run_formal":
        parameters.update(layouts=["horizontal"], bootstrap_replicates=100)
    source = root / "config.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "experiment": {
                    "id": experiment_id or case.experiment_id,
                    "protocol_version": protocol or case.protocol,
                    "parameters": parameters,
                },
                "run": {"seed": 19, "output_root": "output"},
                "tracking": {"mode": "local", "artifacts": "off"},
            }
        )
    )
    return source


@pytest.mark.parametrize("module_name", sorted({case.module for case in CASES}))
def test_canonical_module_imports_in_place_without_gpu_dependencies(module_name):
    qualified_name = f"reladdr.experiments.{module_name}"
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            (
                "import importlib, sys; "
                f"module = importlib.import_module({qualified_name!r}); "
                f"assert module.__name__ == {qualified_name!r}; "
                f"assert sys.modules[{qualified_name!r}] is module; "
                "assert 'torch' not in sys.modules; "
                "assert 'transformers' not in sys.modules"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize("case", PREPARABLE_CASES, ids=lambda case: case.protocol)
def test_prepare_only_records_canonical_identity_without_constructing_model(
    tmp_path, monkeypatch, case
):
    module = importlib.import_module(f"reladdr.experiments.{case.module}")

    def forbidden(*args, **kwargs):
        pytest.fail("prepare-only must not construct a model")

    # 模型类的导入本身不加载 Torch；拦截构造确保离线准备不会进入实际推理。
    importlib.import_module("reladdr.models.qwen2_native")
    monkeypatch.setattr("reladdr.models.qwen2_vl.Qwen2VLAdapter", forbidden)
    monkeypatch.setattr("reladdr.models.qwen2_native.Qwen2NativeAdapter", forbidden)
    source = write_config(tmp_path, case)
    config = load_config(source, parameter_model=module.Parameters)
    directory = getattr(module, case.runner)(config, prepare_only=True)
    record = read_run(directory)
    expected = {"id": case.experiment_id, "protocol_version": case.protocol}
    assert record["metadata"]["experiment"] == expected
    resolved = json.loads((directory / "config.resolved.json").read_text())
    assert {key: resolved["experiment"][key] for key in expected} == expected
    assert record["status"]["status"] == "completed"
    assert not record["metadata"]["details"]["protocol"]["evaluation_executed"]
    assert (directory / "artifacts" / case.artifact).is_file()
    assert not list(directory.glob("artifacts/*/predictions.jsonl"))


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.protocol)
@pytest.mark.parametrize("invalid", ["identity", "protocol"])
def test_wrong_experiment_or_protocol_fails_before_creating_run(tmp_path, case, invalid):
    module = importlib.import_module(f"reladdr.experiments.{case.module}")
    source = write_config(
        tmp_path,
        case,
        experiment_id="Unrelated_Experiment" if invalid == "identity" else None,
        protocol="unrelated-v1" if invalid == "protocol" else None,
    )
    config = load_config(source, parameter_model=module.Parameters)
    kwargs = {} if case.runner == "run_baseline" else {"prepare_only": True}
    with pytest.raises(ValueError):
        getattr(module, case.runner)(config, **kwargs)
    assert not config.document.run.output_root.exists()


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.protocol)
def test_unversioned_historical_identity_is_not_accepted(tmp_path, case):
    module = importlib.import_module(f"reladdr.experiments.{case.module}")
    old_id = {"B01": "E01", "B02": "E02", "M01": "E03", "M02": "E03"}[case.experiment_id[:3]]
    source = write_config(tmp_path, case, experiment_id=old_id)
    config = load_config(source, parameter_model=module.Parameters)
    kwargs = {} if case.runner == "run_baseline" else {"prepare_only": True}
    with pytest.raises(ValueError):
        getattr(module, case.runner)(config, **kwargs)
    assert not config.document.run.output_root.exists()


@pytest.mark.parametrize(
    "command",
    sorted(
        {case.command for case in CASES}
        | {(case.module,) for case in CASES}
        | {("m01_interface_diagnostics",), ("m02_path_diagnostics",)}
    ),
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


@pytest.mark.parametrize("case", PREPARABLE_CASES, ids=lambda case: case.protocol)
def test_real_prepare_cli_records_identity_without_gpu(tmp_path, case):
    source = write_config(tmp_path, case)
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            f"reladdr.experiments.{case.command[0]}",
            *case.command[1:],
            str(source),
            "--prepare-only",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    directory = next((tmp_path / "output").iterdir())
    assert read_run(directory)["metadata"]["experiment"] == {
        "id": case.experiment_id,
        "protocol_version": case.protocol,
    }
    assert not list(directory.glob("artifacts/*/predictions.jsonl"))
