"""验证 E02 私有协议、离线准备及按族统计，不把模拟数值作为模型结果。"""

import json

import pytest
import yaml

from reladdr.experiments.e02 import Parameters, aliases, run_e02, scored, summarize
from reladdr.runtime import ConfigError, load_config, read_run
from reladdr.tasks.patching import PAPER_PALETTE, make_e02_pairs


def config_at(tmp_path, **parameters):
    source = tmp_path / "e02.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "experiment": {
                    "id": "E02",
                    "protocol_version": "e02-v1",
                    "parameters": {
                        "model_path": "no-model",
                        "model_revision": "test-only",
                        "family_counts": {"discovery": 2, "validation": 1, "test": 1},
                        **parameters,
                    },
                },
                "run": {"seed": 19, "output_root": "runs"},
                "tracking": {"mode": "local", "artifacts": "off"},
            }
        )
    )
    return load_config(source, parameter_model=Parameters)


def test_prepare_only_keeps_heldout_and_never_constructs_model(tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("must not load a model")

    root = run_e02(config_at(tmp_path), prepare_only=True, adapter_factory=forbidden)
    assert not read_run(root)["metadata"]["details"]["protocol"]["evaluation_executed"]
    pairs = [
        json.loads(line)
        for line in (root / "artifacts/dataset/pairs.jsonl").read_text().splitlines()
    ]
    assert {p["split"] for p in pairs} == {"discovery", "validation", "test"}
    assert not (root / "artifacts/patching/predictions.jsonl").exists()


def test_score_preserves_full_vocabulary_probability():
    pair = make_e02_pairs(19, {"discovery": 1, "validation": 1, "test": 1}, ["squares_horizontal"])[
        0
    ]
    value = scored(
        {
            "answer_text": pair.address_label.capitalize(),
            "candidate_log_probs": {c: -2.0 for c in PAPER_PALETTE},
        },
        pair,
    )
    assert value["outcome"] == "address"
    assert value["outcome_probabilities"]["address"] == pytest.approx(0.135335283)
    assert len(aliases(PAPER_PALETTE)["red"]) == 4


@pytest.mark.parametrize(
    "parameters",
    [
        {"positions": ["last"]},
        {"layers": [-1]},
        {"image_size": 335},
        {"settings": ["unknown"]},
        {"run_splits": ["test", "test"]},
    ],
)
def test_bad_private_settings_rejected(tmp_path, parameters):
    with pytest.raises(ConfigError):
        config_at(tmp_path, **parameters)


def test_summary_retains_baseline_failure_and_clusters_by_family():
    rows = [
        {
            "phase": "patch",
            "split": "discovery",
            "setting": "squares_horizontal",
            "position": "readout",
            "layer": 20,
            "family_id": f"f{i}",
            "baseline_pair_correct": i == 0,
            "outcome": "address",
            "outcome_probabilities": {"clean": 0.1, "address": 0.8, "attribute": 0.05},
        }
        for i in range(2)
    ]
    summary = summarize(rows, seed=19, replicates=100)
    assert summary["baseline_pair_coverage"] == 0.5
    assert {r["n"] for r in summary["curves"]} == {1, 2}
    all_rows = next(r for r in summary["curves"] if not r["baseline_correct_subset"])
    assert all_rows["outcomes"]["address"]["probability"]["n_families"] == 2
