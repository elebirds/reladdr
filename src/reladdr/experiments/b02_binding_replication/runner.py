"""执行 B02 位置与属性绑定复现，区分原句与紧邻措辞协议。"""

import argparse

from reladdr.experiments._shared.patching import Parameters, run_patching
from reladdr.runtime import load_config
from reladdr.tasks.patching import make_e02_pairs

EXPERIMENT_ID = "B02_Binding_Replication"


def run_replication(config, *, prepare_only=False, adapter_factory=None):
    p = config.parameters
    protocol = config.document.experiment.protocol_version
    if protocol not in {"b02-directional-v1", "b02-adjacent-v1"}:
        raise ValueError("B02 入口要求明确原句或紧邻措辞协议")
    pairs = make_e02_pairs(
        config.document.run.seed,
        p.family_counts,
        p.settings,
        adjacent=protocol == "b02-adjacent-v1",
    )
    return run_patching(
        config,
        pairs,
        experiment_id=EXPERIMENT_ID,
        protocol_version=protocol,
        stage="P1",
        prepare_only=prepare_only,
        adapter_factory=adapter_factory,
    )


def main():
    parser = argparse.ArgumentParser(description="B02 位置/属性绑定复现")
    parser.add_argument("config")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    print(
        run_replication(
            load_config(args.config, parameter_model=Parameters), prepare_only=args.prepare_only
        )
    )


if __name__ == "__main__":
    main()
