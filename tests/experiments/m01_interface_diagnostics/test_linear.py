"""验证 E03 反向配对只按真值可辨识性筛选，保留排除原因和多位置协议。"""

import json

import yaml

from reladdr.experiments.m01_interface_diagnostics.linear import (
    Parameters,
    reverse_identifiable,
    run_e03,
)
from reladdr.runtime import load_config
from reladdr.tasks.patching import make_reference_move_pairs


def test_reverse_pairs_recompute_truth_and_report_ambiguous_labels():
    pairs = make_reference_move_pairs(
        19, {"discovery": 8, "validation": 4, "test": 4}, adjacent=True
    )
    reversed_pairs, excluded = reverse_identifiable(pairs)
    assert len(reversed_pairs) + len(excluded) == len(pairs)
    assert excluded and all(x["reason"] == "reverse_labels_not_distinct" for x in excluded)
    for pair in reversed_pairs:
        assert len({pair.clean_label, pair.address_label, pair.attribute_label}) == 3
        assert pair.clean_label == pair.recipient.answer
        assert pair.attribute_label == pair.donor.answer
        assert pair.address_label == pair.recipient.objects[pair.donor.target_index].color
        assert pair.recipient.prompt == pair.donor.prompt


def test_reverse_prepare_keeps_selection_receipt_and_never_loads_model(tmp_path):
    path = tmp_path / "e03.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "experiment": {
                    "id": "E03",
                    "protocol_version": "e03a-v2",
                    "parameters": {
                        "model_path": "test-only",
                        "model_revision": "test-only",
                        "family_counts": {"discovery": 2, "validation": 1, "test": 1},
                        "positions": ["readout", "text_suffix"],
                        "reverse_pairs": True,
                    },
                },
                "run": {"seed": 19, "output_root": "runs"},
                "tracking": {"mode": "local", "artifacts": "off"},
            }
        )
    )
    directory = run_e03(load_config(path, parameter_model=Parameters), prepare_only=True)
    selection = json.loads((directory / "artifacts/dataset/selection.json").read_text())
    assert selection["reverse_pairs"]
    assert selection["selection_basis"] == "three_distinct_ground_truth_labels_only"
    assert not (directory / "artifacts/patching/predictions.jsonl").exists()
