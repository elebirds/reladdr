"""执行 M02 后缀分组中介，区分充分恢复与必要恢复，兼容 E03b 协议。"""

import argparse
import json
import math
from collections import defaultdict

from pydantic import Field, model_validator

from reladdr.evaluation.clustered import clustered_interval
from reladdr.experiments._shared.patching import Parameters as PatchingParameters
from reladdr.experiments._shared.patching import aliases, scored
from reladdr.interventions.hooks import Site
from reladdr.runtime import RunContext, load_config, resolve_path
from reladdr.tasks.patching import PAPER_PALETTE, make_reference_move_pairs

EXPERIMENT_ID = "M02_Path_Diagnostics"


class Parameters(PatchingParameters):
    layers: list[int] = Field(default_factory=lambda: [16, 17, 18, 19, 20])

    @model_validator(mode="after")
    def check_task_scope(self):
        if self.settings != ["squares_horizontal"]:
            raise ValueError("当前 E03b 只定义 Squares 横排协议")
        if self.positions != ["readout", "question_end"]:
            raise ValueError("E03b 使用固定语义分组，不接受 E02 的 positions 调参")
        return self


def token_groups(prepared):
    suffix = prepared.roles["text_suffix"]
    reference = prepared.roles["reference"]
    user_end = prepared.token_record["user_message_end_index"]
    groups = {
        "vision_end": (suffix[0],),
        "reference": reference,
        "question_tail": tuple(i for i in suffix if reference[-1] < i < user_end),
        "assistant": tuple(i for i in suffix if i >= user_end),
        "relation": prepared.roles["relation"],
    }
    chosen = [i for values in groups.values() for i in values]
    if len(chosen) != len(set(chosen)) or set(chosen) - set(suffix):
        raise ValueError("语义分组必须互不重叠且属于实际后缀")
    groups["other"] = tuple(i for i in suffix if i not in chosen)
    if any(not value for value in groups.values()):
        raise ValueError("本协议要求所有声明分组有实际 token")
    return groups


def mix_group(clean, donor, offsets, *, mode):
    if clean.shape != donor.shape:
        raise ValueError("前后缀端口形状不同")
    if mode == "group_only":
        mixed = clean.clone()
        mixed[list(offsets)] = donor[list(offsets)]
    elif mode == "full_except_group":
        mixed = donor.clone()
        mixed[list(offsets)] = clean[list(offsets)]
    else:
        raise ValueError("未知中介替换类型")
    return mixed


def summarize(rows, seed, replicates):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["split"], row["layer"], row["group"], row["mode"]].append(row)
    output = []
    for (split, layer, group, mode), selected in sorted(grouped.items()):
        for eligible in (False, True):
            subset = [r for r in selected if r["baseline_pair_correct"]] if eligible else selected
            statistics = {}
            for field in (
                "address_probability",
                "address_greedy",
                "address_gain_from_clean",
                "address_change_from_full",
            ):
                totals = defaultdict(lambda: [0.0, 0])
                for row in subset:
                    totals[row["family_id"]][0] += row[field]
                    totals[row["family_id"]][1] += 1
                statistics[field] = clustered_interval(totals, replicates=replicates, seed=seed)
            output.append(
                {
                    "split": split,
                    "layer": layer,
                    "group": group,
                    "mode": mode,
                    "baseline_correct_subset": eligible,
                    "n": len(subset),
                    **statistics,
                }
            )
    return {"resampling_unit": "scene_family", "slices": output, "n_rows": len(rows)}


def run_path(config, *, prepare_only=False):
    p = config.parameters
    if (
        config.document.experiment.id not in {"E03", EXPERIMENT_ID}
        or config.document.experiment.protocol_version != "e03b-mediation-v1"
    ):
        raise ValueError("原生后缀路径入口要求明确 E03b 中介协议")
    pairs = make_reference_move_pairs(config.document.run.seed, p.family_counts, adjacent=True)
    active = [pair for pair in pairs if pair.split in p.run_splits]
    with RunContext(config) as run:
        run.record_metadata(
            "protocol",
            {
                "stage": "P2",
                "step": "E03b token-group mediation",
                "evaluation_executed": not prepare_only,
                "source_result": "20261020T052256208276Z-e03a-suffix-control-c4167205",
                "group_only": "donor group; recipient remaining suffix",
                "full_except_group": "donor suffix; restore recipient group",
                "selection": "all preconstructed pairs; baseline-correct subset separately",
            },
        )
        run.artifacts.write_jsonl(
            "dataset/pairs.jsonl",
            (pair.record() for pair in pairs),
            description="沿用已冻结的同指令参照移位配对",
        )
        if prepare_only:
            return run.run_dir
        from reladdr.models.qwen2_native import Qwen2NativeAdapter

        if p.model_revision.startswith("unverified"):
            raise ValueError("实际运行需要模型来源版本")
        if p.model_source_manifest:
            run.artifacts.write_json(
                "provenance/model_source_files.json",
                json.loads(resolve_path(p.model_source_manifest, config.source.parent).read_text()),
                description="模型及 processor 版本来源",
            )
        with run.time_phase("model_load"):
            adapter = Qwen2NativeAdapter(
                resolve_path(p.model_path, config.source.parent),
                system_prompt=p.system_prompt,
                image_size=p.image_size,
            )
        run.record_metadata("model", {**adapter.metadata, "declared_revision": p.model_revision})
        rows = []
        path = run.artifacts.path_for("mediation/predictions.jsonl")
        with run.time_phase("evaluation"), path.open("x", encoding="utf-8") as stream:

            def save(row):
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                rows.append(row)

            for index, pair in enumerate(active):
                prepared = [
                    adapter.prepare(
                        item.image(),
                        item.prompt,
                        candidate_aliases=aliases(PAPER_PALETTE),
                        reference=item.reference,
                        relation=item.relation,
                    )
                    for item in (pair.recipient, pair.donor)
                ]
                if prepared[0].token_record != prepared[1].token_record:
                    raise ValueError("同指令中介比较要求逐 token 相同的输入布局")
                groups = token_groups(prepared[0])
                sites = [
                    Site(layer, "residual", prepared[0].roles["text_suffix"]) for layer in p.layers
                ]
                clean, clean_cache, _ = adapter.forward(prepared[0], capture=sites)
                donor, donor_cache, _ = adapter.forward(prepared[1], capture=sites)
                baseline_correct = (
                    clean["answer_text"].lower() == pair.recipient.answer
                    and donor["answer_text"].lower() == pair.donor.answer
                )
                baseline_b = math.exp(clean["candidate_log_probs"][pair.address_label])
                base = {
                    "pair_id": pair.pair_id,
                    "family_id": pair.family_id,
                    "split": pair.split,
                    "baseline_pair_correct": baseline_correct,
                    "labels": {
                        "clean": pair.clean_label,
                        "address": pair.address_label,
                        "attribute": pair.attribute_label,
                    },
                }
                name = f"dataset/tokens/{pair.pair_id}.json"
                run.artifacts.write_json(
                    name,
                    {
                        "actual_input": prepared[0].token_record,
                        "groups": {k: list(v) for k, v in groups.items()},
                        "recipient_prediction": clean,
                        "donor_prediction": donor,
                    },
                    description="实际语义分组与未干预基线",
                )
                for site in sites:
                    full, _, _ = adapter.forward(prepared[0], patches={site: donor_cache[site]})
                    full_b = math.exp(full["candidate_log_probs"][pair.address_label])
                    save(
                        {
                            **base,
                            "layer": site.layer,
                            "group": "all_suffix",
                            "mode": "full",
                            "address_probability": full_b,
                            "address_greedy": full["answer_text"].lower() == pair.address_label,
                            "address_gain_from_clean": full_b - baseline_b,
                            "address_change_from_full": 0.0,
                            **scored(full, pair),
                        }
                    )
                    for group, indices in groups.items():
                        offsets = [site.positions.index(i) for i in indices]
                        for mode in ("group_only", "full_except_group"):
                            mixed = mix_group(
                                clean_cache[site], donor_cache[site], offsets, mode=mode
                            )
                            pred, _, _ = adapter.forward(prepared[0], patches={site: mixed})
                            b = math.exp(pred["candidate_log_probs"][pair.address_label])
                            save(
                                {
                                    **base,
                                    "layer": site.layer,
                                    "group": group,
                                    "mode": mode,
                                    "physical_indices": list(indices),
                                    "address_probability": b,
                                    "address_greedy": pred["answer_text"].lower()
                                    == pair.address_label,
                                    "address_gain_from_clean": b - baseline_b,
                                    "address_change_from_full": b - full_b,
                                    **scored(pred, pair),
                                }
                            )
                run.logger.info(
                    "E03b 中介 %s/%s，基线正确=%s", index + 1, len(active), baseline_correct
                )
        run.artifacts.register(
            "mediation/predictions.jsonl", description="充分/必要两类分组中介的逐对真实输出"
        )
        run.artifacts.write_json(
            "mediation/summary.json",
            summarize(rows, config.document.run.seed, p.bootstrap_replicates),
            description="全量与基线正确子集的场景族区间",
        )
        run.record_metadata(
            "execution",
            {
                "native_forward_calls": adapter.forward_calls,
                "n_pairs": len(active),
                "evaluated_splits": p.run_splits,
            },
        )
        run.record_metadata("gpu_memory", adapter.peak_memory())
    return run.run_dir


def main():
    parser = argparse.ArgumentParser(description="M02 后缀分组中介（兼容 E03b mediation 协议）")
    parser.add_argument("config")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    print(
        run_path(
            load_config(args.config, parameter_model=Parameters), prepare_only=args.prepare_only
        )
    )


if __name__ == "__main__":
    main()
