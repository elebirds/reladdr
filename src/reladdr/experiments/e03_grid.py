"""执行二维同指令参照移位，排除地址标签与参照颜色重叠的实验混淆。"""

import argparse

from pydantic import Field, model_validator

from reladdr.experiments.e02 import Parameters as PatchingParameters
from reladdr.experiments.e02 import run_patching
from reladdr.runtime import load_config
from reladdr.tasks.patching import make_grid_move_pairs


class Parameters(PatchingParameters):
    settings: list[str] = Field(default_factory=lambda: ["squares_grid2"])

    @model_validator(mode="after")
    def check_scope(self):
        if self.settings != ["squares_grid2"]:
            raise ValueError("二维辨识控制只接受 squares_grid2")
        return self


def run_grid(config, *, prepare_only=False, adapter_factory=None):
    pairs = make_grid_move_pairs(config.document.run.seed, config.parameters.family_counts)
    return run_patching(
        config,
        pairs,
        experiment_id="E03",
        protocol_version="e03a-grid-v1",
        stage="P2",
        prepare_only=prepare_only,
        adapter_factory=adapter_factory,
        full_state_check="text_suffix" in config.parameters.positions,
        dataset_notes={
            "address_label_differs_from_reference_color": True,
            "same_instruction": True,
            "same_geometric_slots": True,
            "directions": ["left", "right", "above", "below"],
        },
    )


def main():
    parser = argparse.ArgumentParser(description="E03a 二维辨识控制")
    parser.add_argument("config")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    print(
        run_grid(
            load_config(args.config, parameter_model=Parameters), prepare_only=args.prepare_only
        )
    )


if __name__ == "__main__":
    main()
