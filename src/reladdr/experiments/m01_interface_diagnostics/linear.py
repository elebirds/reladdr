"""执行 M01 一维接口诊断：相同语言指令下改变图中参照位置，兼容 E03a。"""

import argparse
from dataclasses import replace

from pydantic import model_validator

from reladdr.experiments._shared.patching import Parameters as PatchingParameters
from reladdr.experiments._shared.patching import run_patching
from reladdr.runtime import load_config
from reladdr.tasks.patching import make_reference_move_pairs

EXPERIMENT_ID = "M01_Interface_Diagnostics"


class Parameters(PatchingParameters):
    reverse_pairs: bool = False

    @model_validator(mode="after")
    def check_task_scope(self):
        if self.settings != ["squares_horizontal"]:
            raise ValueError("当前 E03a 只定义 Squares 横排协议，不能静默切换 setting")
        return self


def reverse_identifiable(pairs):
    retained = []
    excluded = []
    for pair in pairs:
        labels = (
            pair.donor.answer,
            pair.donor.objects[pair.recipient.target_index].color,
            pair.recipient.answer,
        )
        if len(set(labels)) < 3:
            excluded.append(
                {
                    "pair_id": pair.pair_id,
                    "split": pair.split,
                    "reason": "reverse_labels_not_distinct",
                    "labels": labels,
                }
            )
            continue
        retained.append(
            replace(
                pair,
                pair_id=pair.pair_id + "-reverse",
                recipient=pair.donor,
                donor=pair.recipient,
                clean_label=labels[0],
                address_label=labels[1],
                attribute_label=labels[2],
                contrast=pair.contrast + "_reverse",
            )
        )
    return retained, excluded


def run_e03(config, *, prepare_only=False, adapter_factory=None):
    protocol = config.document.experiment.protocol_version
    if protocol not in {"e03a-v1", "e03a-v2"}:
        raise ValueError("E03a 入口要求明确 v1 directional 或 v2 adjacent 协议")
    pairs = make_reference_move_pairs(
        config.document.run.seed, config.parameters.family_counts, adjacent=protocol == "e03a-v2"
    )
    excluded = []
    if config.parameters.reverse_pairs:
        pairs, excluded = reverse_identifiable(pairs)
    return run_patching(
        config,
        pairs,
        experiment_id=EXPERIMENT_ID,
        compatible_experiment_ids=("E03",),
        protocol_version=protocol,
        stage="P2",
        prepare_only=prepare_only,
        adapter_factory=adapter_factory,
        dataset_notes={
            "reverse_pairs": config.parameters.reverse_pairs,
            "excluded": excluded,
            "selection_basis": "three_distinct_ground_truth_labels_only",
        },
        full_state_check="text_suffix" in config.parameters.positions,
    )


def main():
    parser = argparse.ArgumentParser(description="M01 一维同指令参照移位（兼容 E03a）")
    parser.add_argument("config")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    print(
        run_e03(
            load_config(args.config, parameter_model=Parameters), prepare_only=args.prepare_only
        )
    )


if __name__ == "__main__":
    main()
