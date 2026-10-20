"""执行 M02 条件写入冻结，检验新关系/参照对在线组件变化的依赖。"""

import argparse
import json
import math
from collections import defaultdict
from dataclasses import replace

from pydantic import Field, model_validator

from reladdr.evaluation.clustered import clustered_interval
from reladdr.experiments._shared.patching import Parameters as PatchingParameters
from reladdr.experiments._shared.patching import aliases
from reladdr.interventions.hooks import Site
from reladdr.runtime import RunContext, load_config, resolve_path
from reladdr.tasks.patching import (
    OPPOSITE,
    PAPER_PALETTE,
    PREPOSITIONS,
    make_e02_pairs,
    make_grid_move_pairs,
)

EXPERIMENT_ID = "M02_Path_Diagnostics"


class Parameters(PatchingParameters):
    settings: list[str] = Field(default_factory=lambda: ["squares_horizontal", "squares_grid2"])
    selection_layers: list[int] = Field(default_factory=lambda: list(range(16, 23)))

    @model_validator(mode="after")
    def check_layers(self):
        if set(self.settings) - {"squares_horizontal", "squares_grid2"}:
            raise ValueError("组件冻结仅定义关系变化横排和参照变化二维格")
        if not self.selection_layers or set(self.selection_layers) - set(self.layers):
            raise ValueError("选择窗口必须属于实际扫描层")
        return self


def condition_pairs(seed, counts):
    rows = []
    seen = set()
    for pair in make_e02_pairs(seed, counts, ["squares_horizontal"], adjacent=True):
        key = (pair.family_id, pair.recipient.relation)
        if key in seen:
            continue
        seen.add(key)
        old = pair.recipient
        relation = OPPOSITE[old.relation]
        target = 2 - old.target_index
        new = replace(
            old,
            input_id=old.input_id + "-opposite",
            relation=relation,
            target_index=target,
            answer=old.objects[target].color,
            prompt=old.prompt.replace(PREPOSITIONS[old.relation], PREPOSITIONS[relation]),
        )
        rows.append((pair.family_id + "-condition-" + old.relation, "relation_only", old, new))
    for pair in make_grid_move_pairs(seed, counts):
        old = pair.recipient
        delta = {"left": -1, "right": 1, "above": -2, "below": 2}[old.relation]
        reference_index = pair.donor.target_index - delta
        reference = f"{old.objects[reference_index].color} square"
        new = replace(
            old,
            input_id=old.input_id + "-new-reference",
            reference=reference,
            target_index=pair.donor.target_index,
            answer=pair.address_label,
            prompt=old.prompt.replace(old.reference, reference),
        )
        rows.append((pair.pair_id + "-same-image", "reference_only", old, new))
    return rows


def summarize(rows, seed, replicates):
    groups = defaultdict(list)
    for row in rows:
        groups[row["split"], row["contrast"], row["mode"]].append(row)
    output = []
    for (split, contrast, mode), selected in sorted(groups.items()):
        for eligible in (False, True):
            subset = [r for r in selected if r["baseline_pair_correct"]] if eligible else selected
            fields = {}
            for field in (
                "expected_correct",
                "expected_probability",
                "probability_change_from_natural",
            ):
                totals = defaultdict(lambda: [0.0, 0])
                for row in subset:
                    totals[row["family_id"]][0] += row[field]
                    totals[row["family_id"]][1] += 1
                fields[field] = clustered_interval(totals, replicates=replicates, seed=seed)
            output.append(
                {
                    "split": split,
                    "contrast": contrast,
                    "mode": mode,
                    "baseline_correct_subset": eligible,
                    "n": len(subset),
                    **fields,
                }
            )
    return {"slices": output, "resampling_unit": "scene_family", "n_rows": len(rows)}


def run_freeze(config, *, prepare_only=False):
    p = config.parameters
    if (
        config.document.experiment.id != EXPERIMENT_ID
        or config.document.experiment.protocol_version != "m02-conditional-freeze-v1"
    ):
        raise ValueError("要求明确在线条件写入冻结协议")
    cases = [
        case
        for case in condition_pairs(config.document.run.seed, p.family_counts)
        if case[2].setting in p.settings
    ]
    active = [row for row in cases if row[2].split in p.run_splits]
    with RunContext(config) as run:
        run.record_metadata(
            "protocol",
            {
                "stage": "P2",
                "evaluation_executed": not prepare_only,
                "contrasts": ["same_image_change_relation", "same_image_change_reference"],
                "component_freeze": (
                    "new natural question; declared text writes are fixed to old-condition values"
                ),
                "text_scope": "after vision_end; all question and assistant tokens",
                "selection_layers": p.selection_layers,
            },
        )
        run.artifacts.write_jsonl(
            "dataset/cases.jsonl",
            (
                {"case_id": key, "contrast": contrast, "old": old.record(), "new": new.record()}
                for key, contrast, old, new in cases
            ),
            description="自然条件变化，图像完全相同",
        )
        if prepare_only:
            return run.run_dir
        from reladdr.models.qwen2_native import Qwen2NativeAdapter

        if p.model_revision.startswith("unverified"):
            raise ValueError("实际运行需要模型版本")
        if p.model_source_manifest:
            run.artifacts.write_json(
                "provenance/model_source_files.json",
                json.loads(resolve_path(p.model_source_manifest, config.source.parent).read_text()),
                description="模型与 processor 来源",
            )
        with run.time_phase("model_load"):
            adapter = Qwen2NativeAdapter(
                resolve_path(p.model_path, config.source.parent),
                system_prompt=p.system_prompt,
                image_size=p.image_size,
            )
        run.record_metadata("model", {**adapter.metadata, "declared_revision": p.model_revision})
        rows = []
        path = run.artifacts.path_for("freezing/predictions.jsonl")
        with run.time_phase("evaluation"), path.open("x", encoding="utf-8") as stream:
            for index, (key, contrast, old, new) in enumerate(active):
                prepared = [
                    adapter.prepare(
                        old.image(),
                        item.prompt,
                        candidate_aliases=aliases(PAPER_PALETTE),
                        reference=item.reference,
                        relation=item.relation,
                    )
                    for item in (old, new)
                ]
                positions = prepared[0].roles["text_suffix"][1:]
                if positions != prepared[1].roles["text_suffix"][1:]:
                    raise ValueError("冻结条件要求位置逐一对齐，不自动广播或补齐")
                sites = [
                    Site(layer, component, positions)
                    for layer in p.layers
                    for component in ("mlp", "attention")
                ]
                old_pred, cache, old_logits = adapter.forward(prepared[0], capture=sites)
                natural, _, _ = adapter.forward(prepared[1])
                good = (
                    old_pred["answer_text"].lower() == old.answer
                    and natural["answer_text"].lower() == new.answer
                )
                natural_probability = math.exp(natural["candidate_log_probs"][new.answer])
                if index == 0:
                    _, _, noop_logits = adapter.forward(prepared[0], patches=cache)
                    error = float((noop_logits - old_logits).abs().max().item())
                    run.record_metadata("all_component_noop", {"max_abs_logit_difference": error})
                    if error > 0.01:
                        raise RuntimeError("全组件自身替换改变 logits")
                modes = {
                    "freeze_all_text_mlp": {
                        site: value for site, value in cache.items() if site.component == "mlp"
                    },
                    "freeze_selection_mlp": {
                        site: value
                        for site, value in cache.items()
                        if site.component == "mlp" and site.layer in p.selection_layers
                    },
                    "freeze_all_text_attention": {
                        site: value
                        for site, value in cache.items()
                        if site.component == "attention"
                    },
                }
                results = {"natural": natural}
                for mode, patches in modes.items():
                    results[mode] = adapter.forward(prepared[1], patches=patches)[0]
                for mode, pred in results.items():
                    probability = math.exp(pred["candidate_log_probs"][new.answer])
                    row = {
                        "case_id": key,
                        "family_id": old.family_id,
                        "split": old.split,
                        "contrast": contrast,
                        "mode": mode,
                        "old_answer": old.answer,
                        "expected_answer": new.answer,
                        "baseline_pair_correct": good,
                        "expected_correct": pred["answer_text"].lower() == new.answer,
                        "expected_probability": probability,
                        "probability_change_from_natural": probability - natural_probability,
                        **pred,
                    }
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    rows.append(row)
                run.artifacts.write_json(
                    "dataset/tokens/" + key + ".json",
                    {"old": prepared[0].token_record, "new": prepared[1].token_record},
                    description="实际 token 与冻结范围",
                )
                run.logger.info(
                    "M02 在线写入冻结 %s/%s，%s，基线正确=%s",
                    index + 1,
                    len(active),
                    contrast,
                    good,
                )
        run.artifacts.register(
            "freezing/predictions.jsonl", description="条件变化与组件写入冻结的真实输出"
        )
        run.artifacts.write_json(
            "freezing/summary.json",
            summarize(rows, config.document.run.seed, p.bootstrap_replicates),
            description="按族区间、全量和基线正确子集",
        )
        run.record_metadata(
            "execution",
            {
                "native_forward_calls": adapter.forward_calls,
                "n_cases": len(active),
                "evaluated_splits": p.run_splits,
            },
        )
        run.record_metadata("gpu_memory", adapter.peak_memory())
    return run.run_dir


def main():
    parser = argparse.ArgumentParser(description="M02 条件写入冻结")
    parser.add_argument("config")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    print(
        run_freeze(
            load_config(args.config, parameter_model=Parameters), prepare_only=args.prepare_only
        )
    )


if __name__ == "__main__":
    main()
