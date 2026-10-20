"""用明确标注的离线适配器验证正式 E01 的划分、记录、统计和故障保存，不伪造 GPU 证据。"""

import json

import pytest
import yaml

from reladdr.evaluation.clustered import clustered_interval
from reladdr.experiments.e01_formal import Parameters, run_formal
from reladdr.runtime import ConfigError, load_config, read_run
from reladdr.tasks.synthetic import PALETTE


class ControlledAdapter:
    metadata = {"kind": "controlled-test-client", "model_loaded": False}

    def __init__(self, model_path, *, max_new_tokens, system_prompt):
        pass

    def predict(self, image, prompt, *, candidate_aliases):
        return {
            "answer_text": "red",
            "candidate_log_probs": {color: -1.0 if color == "red" else -4.0 for color in PALETTE},
        }

    def peak_memory(self):
        return {"max_allocated_bytes": 0}


def make_config(tmp_path, **parameters):
    path = tmp_path / "formal.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "experiment": {
                    "id": "E01",
                    "protocol_version": "e01-v1",
                    "parameters": {
                        "model_path": "controlled-model",
                        "model_revision": "controlled-test-only",
                        "family_counts": {"discovery": 2, "validation": 1, "test": 1},
                        "layouts": ["horizontal", "grid3"],
                        "bootstrap_replicates": 100,
                        **parameters,
                    },
                },
                "run": {"seed": 19, "output_root": "runs"},
                "tracking": {"mode": "local", "artifacts": "off"},
            }
        )
    )
    return load_config(path, parameter_model=Parameters)


def test_prepare_never_loads_adapter_and_reserves_validation_test(tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("prepare-only must not load a model")

    directory = run_formal(make_config(tmp_path), adapter_factory=forbidden, prepare_only=True)
    record = read_run(directory)
    assert record["status"]["status"] == "completed"
    assert not record["metadata"]["details"]["protocol"]["evaluation_executed"]
    manifest = json.loads((directory / "artifacts/dataset/manifest.json").read_text())
    assert manifest["evaluated_splits"] == []
    assert manifest["planned_model_calls"] > 0
    assert not (directory / "artifacts/baseline/predictions.jsonl").exists()


def test_formal_run_keeps_actual_labels_and_family_level_statistics(tmp_path):
    directory = run_formal(make_config(tmp_path), adapter_factory=ControlledAdapter)
    rows = [
        json.loads(line)
        for line in (directory / "artifacts/baseline/predictions.jsonl").read_text().splitlines()
    ]
    assert {row["split"] for row in rows} == {"discovery"}
    assert len({row["family_id"] for row in rows}) == 2
    for row in rows:
        assert (directory / row["image_path"]).is_file()
        assert row["fixed_foil"] == row["foil"]
        if row["condition"] == "blank":
            assert row["correct"] is None
        elif row["condition"] != "original":
            assert row["displayed_answer"] != row["answer"]
    summary = json.loads((directory / "artifacts/baseline/summary.json").read_text())
    assert summary["n_predictions"] == len(rows)
    assert summary["resampling_unit"] == "scene_family"
    assert summary["evaluated_splits"] == ["discovery"]
    for row in summary["slices"]:
        assert row["accuracy"]["n_families"] == (0 if row["condition"] == "blank" else 2)
    assert {row["contrast"] for row in summary["paired"]} >= {
        "target_recolor",
        "position_swap",
        "reference_change",
        "relation_change",
    }
    assert summary["diagnostic_patterns"]
    assert read_run(directory)["tracking"]["status"] == "disabled"


def test_cluster_bootstrap_uses_family_count_not_question_count():
    result = clustered_interval(
        {"family-a": (100.0, 100), "family-b": (0.0, 1)}, replicates=500, seed=19
    )
    assert result["n_families"] == 2
    assert result["n"] == 101
    assert result["estimate"] == 100 / 101
    assert result["ci95"] == [0.0, 1.0]
    assert clustered_interval({"a": (1.0, 2)}, replicates=100, seed=1)["ci95"] is None


def test_failed_gpu_call_preserves_already_written_predictions(tmp_path):
    class FailingAdapter(ControlledAdapter):
        calls = 0

        def predict(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("controlled inference failure")
            return super().predict(*args, **kwargs)

    config = make_config(tmp_path, layouts=["horizontal"])
    with pytest.raises(RuntimeError, match="controlled inference failure"):
        run_formal(config, adapter_factory=FailingAdapter)
    directory = next(config.document.run.output_root.iterdir())
    assert read_run(directory)["status"]["status"] == "failed"
    assert len((directory / "artifacts/baseline/predictions.jsonl").read_text().splitlines()) == 1
    assert not (directory / "artifacts/baseline/summary.json").exists()


@pytest.mark.parametrize(
    "parameters",
    [
        {"conditions": ["blank"]},
        {"layouts": ["grid3", "grid3"]},
        {"run_splits": ["debug"]},
        {"templates": ["unknown"]},
    ],
)
def test_invalid_private_protocol_is_rejected(tmp_path, parameters):
    with pytest.raises(ConfigError):
        make_config(tmp_path, **parameters)


def test_model_file_manifest_is_preserved_with_declared_version(tmp_path):
    source = tmp_path / "source-files.json"
    original = [{"Path": "config.json", "Revision": "controlled-file-revision"}]
    source.write_text(json.dumps(original))
    config = make_config(
        tmp_path,
        layouts=["horizontal"],
        conditions=["original"],
        model_source_manifest=str(source),
    )
    directory = run_formal(config, adapter_factory=ControlledAdapter)
    assert (
        json.loads((directory / "artifacts/provenance/model_source_files.json").read_text())
        == original
    )
    assert read_run(directory)["metadata"]["details"]["model"]["declared_revision"] == (
        "controlled-test-only"
    )
