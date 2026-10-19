"""以受控适配器验证 E01 记录闭环；GPU 结果必须由真实服务器另行取得。"""

import json
from pathlib import Path

import pytest

from reladdr.experiments.e01 import Parameters, run_baseline, score_prediction
from reladdr.models.qwen2_vl import candidate_token_ids
from reladdr.runtime import ConfigError, load_config, read_run
from reladdr.tasks.synthetic import PALETTE


class ControlledAdapter:
    metadata = {"kind": "controlled-test-client", "model_loaded": False}

    def __init__(self, model_path, *, max_new_tokens):
        self.model_path = model_path

    def predict(self, image, prompt, *, candidate_aliases=None):
        return {
            "answer_text": "red",
            "candidate_log_probs": {color: -1.0 if color == "red" else -4.0 for color in PALETTE},
        }

    def peak_memory(self):
        return {"max_allocated_bytes": 0, "max_reserved_bytes": 0}


def make_config(tmp_path, conditions=None):
    import yaml

    path = tmp_path / "e01.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "experiment": {
                    "id": "E01_smoke",
                    "protocol_version": "1",
                    "parameters": {
                        "model_path": "models/local",
                        "model_revision": "controlled-test-only",
                        "image_conditions": conditions or ["original", "blank", "mismatch"],
                    },
                },
                "run": {"seed": 19, "output_root": "runs"},
                "tracking": {"mode": "local", "artifacts": "off"},
            }
        )
    )
    return load_config(path, parameter_model=Parameters)


def test_e01_records_images_truth_outputs_and_separate_control_semantics(tmp_path):
    config = make_config(tmp_path)
    directory = run_baseline(config, adapter_factory=ControlledAdapter)
    record = read_run(directory)
    assert record["status"]["status"] == "completed"
    assert record["metadata"]["details"]["protocol"]["split"] == "debug"
    rows = [
        json.loads(line)
        for line in (directory / "artifacts/baseline/predictions.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 60
    for row in rows:
        assert (directory / row["image_path"]).is_file()
        if row["condition"] == "blank":
            assert row["displayed_answer"] is None
            assert row["correct"] is None
        elif row["condition"] == "mismatch":
            assert row["displayed_answer"] != row["answer"]
    summary = json.loads((directory / "artifacts/baseline/summary.json").read_text())
    assert "accuracy" not in summary["blank.relation"]
    assert "accuracy" in summary["mismatch.relation"]
    assert record["tracking"]["status"] == "disabled"


def test_prediction_format_errors_are_kept_separate():
    prediction = ControlledAdapter(Path("."), max_new_tokens=8).predict(None, "")
    assert score_prediction(prediction, "red")["format_valid"]
    prediction["answer_text"] = "The color is red."
    scored = score_prediction(prediction, "red")
    assert not scored["format_valid"]
    assert scored["candidate_prediction"] == "red"


def test_private_conditions_reject_unknown_mode(tmp_path):
    with pytest.raises(ConfigError):
        make_config(tmp_path, ["original", "unknown"])


def test_gpu_failure_preserves_local_truth_and_failed_status(tmp_path):
    def unavailable(*args, **kwargs):
        raise RuntimeError("controlled GPU unavailable")

    config = make_config(tmp_path)
    with pytest.raises(RuntimeError, match="GPU unavailable"):
        run_baseline(config, adapter_factory=unavailable)
    directories = list(config.document.run.output_root.iterdir())
    assert len(directories) == 1
    assert read_run(directories[0])["status"]["status"] == "failed"
    assert (directories[0] / "artifacts/dataset/scenes.jsonl").is_file()


def test_multi_token_candidate_cannot_be_truncated():
    class Tokenizer:
        def encode(self, text, **kwargs):
            return [1] if text == "prefix" else [1, 2, 3]

    with pytest.raises(ValueError, match="单个 token"):
        candidate_token_ids(Tokenizer(), "prefix", {"red": ["red", "Red"]})
