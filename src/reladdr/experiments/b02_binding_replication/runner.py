"""执行 B02 位置与属性绑定复现，兼容 E02 的原句与紧邻措辞协议。"""

import argparse

from reladdr.experiments._shared.patching import Parameters as Parameters
from reladdr.experiments._shared.patching import aliases as aliases
from reladdr.experiments._shared.patching import run_patching as run_patching
from reladdr.experiments._shared.patching import scored as scored
from reladdr.experiments._shared.patching import summarize as summarize
from reladdr.runtime import load_config
from reladdr.tasks.patching import make_e02_pairs

EXPERIMENT_ID = "B02_Binding_Replication"


def run_e02(config, *, prepare_only=False, adapter_factory=None):
    p = config.parameters
    protocol = config.document.experiment.protocol_version
    if protocol not in {"e02-v1", "e02-adjacent-v1"}:
        raise ValueError("E02 入口要求明确原句或紧邻措辞协议")
    pairs = make_e02_pairs(
        config.document.run.seed,
        p.family_counts,
        p.settings,
        adjacent=protocol == "e02-adjacent-v1",
    )
    return run_patching(
        config,
        pairs,
        experiment_id=EXPERIMENT_ID,
        compatible_experiment_ids=("E02",),
        protocol_version=protocol,
        stage="P1",
        prepare_only=prepare_only,
        adapter_factory=adapter_factory,
    )


def main():
    parser = argparse.ArgumentParser(description="B02 位置/属性绑定复现（兼容 E02 协议）")
    parser.add_argument("config")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    print(
        run_e02(
            load_config(args.config, parameter_model=Parameters), prepare_only=args.prepare_only
        )
    )


if __name__ == "__main__":
    main()
