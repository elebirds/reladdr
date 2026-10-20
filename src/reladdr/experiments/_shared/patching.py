"""共享三标签原生移植的参数、评分、执行与记录；各子包持有自己的科学协议。"""

import json
import math
from collections import defaultdict

from pydantic import Field, model_validator

from reladdr.evaluation.clustered import clustered_interval
from reladdr.interventions.hooks import Site
from reladdr.runtime import RunContext, resolve_path
from reladdr.runtime.config import StrictModel
from reladdr.tasks.patching import PAPER_PALETTE
from reladdr.tasks.synthetic import PALETTE


class Parameters(StrictModel):
    model_path: str
    model_revision: str
    model_source_manifest: str | None = None
    system_prompt: str = "You are a helpful assistant. You respond in one token."
    image_size: int = Field(default=336, ge=28)
    family_counts: dict[str, int] = Field(
        default_factory=lambda: {"discovery": 8, "validation": 4, "test": 4}
    )
    run_splits: list[str] = Field(default_factory=lambda: ["discovery"])
    settings: list[str] = Field(default_factory=lambda: ["squares_horizontal"])
    positions: list[str] = Field(default_factory=lambda: ["readout", "question_end"])
    layers: list[int] = Field(default_factory=lambda: list(range(28)))
    bootstrap_replicates: int = Field(default=1000, ge=100)

    @model_validator(mode="after")
    def check_protocol(self):
        for name, allowed in (
            ("run_splits", {"discovery", "validation", "test"}),
            (
                "settings",
                {"squares_horizontal", "shapes_horizontal", "shapes_vertical", "squares_grid2"},
            ),
            ("positions", {"readout", "question_end", "text_suffix"}),
        ):
            values = getattr(self, name)
            if not values or len(values) != len(set(values)) or set(values) - allowed:
                raise ValueError(f"{name} 取值不合法")
        if not self.layers or min(self.layers) < 0 or len(set(self.layers)) != len(self.layers):
            raise ValueError("层编号必须非负、不重复、非空")
        if self.image_size % 28:
            raise ValueError("视觉边长必须为 28 的整数倍")
        return self


def aliases(palette):
    # R3 源码的四种拼写概率聚合；B01 使用独立的双拼写协议。
    return {
        color: [color, color.capitalize(), " " + color, " " + color.capitalize()]
        for color in palette
    }


def scored(prediction, pair):
    labels = {
        "clean": pair.clean_label,
        "address": pair.address_label,
        "attribute": pair.attribute_label,
    }
    return {
        **prediction,
        "outcome": next(
            (
                name
                for name, label in labels.items()
                if prediction["answer_text"].strip().lower() == label
            ),
            "other",
        ),
        "outcome_probabilities": {
            name: math.exp(prediction["candidate_log_probs"][label])
            for name, label in labels.items()
        },
    }


def summarize(rows, *, seed, replicates):
    groups = defaultdict(list)
    for row in rows:
        if row["phase"] == "patch":
            groups[row["split"], row["setting"], row["position"], row["layer"]].append(row)
    output = []
    for (split, setting, position, layer), selected in sorted(groups.items()):
        for eligible_only in (False, True):
            subset = (
                [r for r in selected if r["baseline_pair_correct"]] if eligible_only else selected
            )
            fields = {}
            for outcome in ("clean", "address", "attribute"):
                totals = defaultdict(lambda: [0.0, 0])
                counts = defaultdict(lambda: [0.0, 0])
                for row in subset:
                    totals[row["family_id"]][0] += row["outcome_probabilities"][outcome]
                    totals[row["family_id"]][1] += 1
                    counts[row["family_id"]][0] += row["outcome"] == outcome
                    counts[row["family_id"]][1] += 1
                fields[outcome] = {
                    "probability": clustered_interval(totals, replicates=replicates, seed=seed),
                    "greedy_rate": clustered_interval(counts, replicates=replicates, seed=seed),
                }
            output.append(
                {
                    "split": split,
                    "setting": setting,
                    "position": position,
                    "layer": layer,
                    "baseline_correct_subset": eligible_only,
                    "n": len(subset),
                    "outcomes": fields,
                }
            )
    return {
        "n_rows": len(rows),
        "curves": output,
        "resampling_unit": "scene_family",
        "baseline_pair_coverage": sum(
            r["baseline_pair_correct"] for r in rows if r["phase"] == "patch"
        )
        / max(1, sum(r["phase"] == "patch" for r in rows)),
    }


def run_patching(
    config,
    pairs,
    *,
    experiment_id,
    protocol_version,
    stage,
    prepare_only=False,
    adapter_factory=None,
    dataset_notes=None,
    full_state_check=False,
):
    """共用三标签替换和落盘；样本、实验编号及解释仍由各自协议拥有。"""
    p = config.parameters
    if (
        config.document.experiment.id != experiment_id
        or config.document.experiment.protocol_version != protocol_version
    ):
        raise ValueError("实验入口与声明协议不匹配")
    active = [pair for pair in pairs if pair.split in p.run_splits]
    with RunContext(config) as run:
        run.record_metadata(
            "protocol",
            {
                "stage": stage,
                "evaluation_executed": not prepare_only,
                "positions": p.positions,
                "layer_convention": "zero_based_decoder_block_post",
                "upstream": (
                    "Nix07/spatial-variable-binding/last_token_exp/src/patch.py; "
                    "final physical token"
                ),
                "adaptations": [
                    "HF native hooks instead of vendored TransformerLens",
                    "batch=1 BF16 SDPA",
                    "family-grouped fresh pairs; all data and correctness coverage reported",
                ],
            },
        )
        run.artifacts.write_jsonl(
            "dataset/pairs.jsonl",
            (pair.record() for pair in pairs),
            description="固定族划分及三个可区分干预结果标签",
        )
        if dataset_notes is not None:
            run.artifacts.write_json(
                "dataset/selection.json",
                dataset_notes,
                description="仅按真值可区分性构造/排除的样本，非按模型表现筛选",
            )
        saved = set()
        for pair in active:
            for item in (pair.recipient, pair.donor):
                name = f"dataset/images/{item.input_id}.png"
                if name not in saved:
                    item.image().save(run.artifacts.path_for(name))
                    run.artifacts.register(name, description="实际原图或 donor 图")
                    saved.add(name)
        if prepare_only:
            return run.run_dir
        if adapter_factory is None:
            if p.model_revision.startswith("unverified"):
                raise ValueError("实际执行必须填写模型来源版本")
            from reladdr.models.qwen2_native import Qwen2NativeAdapter

            adapter_factory = Qwen2NativeAdapter
        if p.model_source_manifest:
            run.artifacts.write_json(
                "provenance/model_source_files.json",
                json.loads(resolve_path(p.model_source_manifest, config.source.parent).read_text()),
                description="模型与 processor 逐文件来源",
            )
        with run.time_phase("model_load"):
            adapter = adapter_factory(
                resolve_path(p.model_path, config.source.parent),
                system_prompt=p.system_prompt,
                image_size=p.image_size,
            )
        run.record_metadata("model", {**adapter.metadata, "declared_revision": p.model_revision})
        if max(p.layers) >= adapter.n_layers:
            raise ValueError("层编号超出实际模型")
        rows = []
        path = run.artifacts.path_for("patching/predictions.jsonl")

        def save(stream, row):
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            rows.append(row)

        with run.time_phase("evaluation"), path.open("x", encoding="utf-8") as stream:
            for index, pair in enumerate(active):
                palette = PAPER_PALETTE if pair.recipient.palette == "paper" else PALETTE
                prepared = [
                    adapter.prepare(
                        item.image(),
                        item.prompt,
                        candidate_aliases=aliases(palette),
                        reference=item.reference,
                        relation=item.relation,
                    )
                    for item in (pair.recipient, pair.donor)
                ]
                sites = [
                    [
                        Site(layer, "residual", item.roles[position])
                        for position in p.positions
                        for layer in p.layers
                    ]
                    for item in prepared
                ]
                base = []
                caches = []
                logits = []
                for item, locations in zip(prepared, sites, strict=True):
                    pred, cache, values = adapter.forward(item, capture=locations)
                    base.append(pred)
                    caches.append(cache)
                    logits.append(values)
                if index == 0:
                    legacy = adapter.predict(
                        pair.recipient.image(),
                        pair.recipient.prompt,
                        candidate_aliases=aliases(palette),
                    )
                    legacy_error = max(
                        abs(legacy["candidate_log_probs"][c] - base[0]["candidate_log_probs"][c])
                        for c in palette
                    )
                    run.record_metadata(
                        "readout_equivalence",
                        {
                            "max_candidate_log_probability_difference": legacy_error,
                            "legacy_answer": legacy["answer_text"],
                            "native_answer": base[0]["answer_text"],
                            "additional_generate_calls": 1,
                        },
                    )
                    if (
                        legacy_error > 0.01
                        or legacy["answer_text"].strip().lower()
                        != base[0]["answer_text"].strip().lower()
                    ):
                        raise RuntimeError("原生前向读出与原有生成入口不一致")
                if index == 0 and full_state_check:
                    if (
                        prepared[0].token_record["input_tokens"]
                        != prepared[1].token_record["input_tokens"]
                        or pair.recipient.prompt != pair.donor.prompt
                    ):
                        raise ValueError("完整状态恢复检查要求相同问题和序列布局")
                    clean_all = Site(
                        p.layers[0],
                        "residual",
                        tuple(range(prepared[0].token_record["input_tokens"])),
                    )
                    donor_all = Site(
                        p.layers[0],
                        "residual",
                        tuple(range(prepared[1].token_record["input_tokens"])),
                    )
                    _, full_cache, _ = adapter.forward(prepared[1], capture=[donor_all])
                    restored_donor, _, full_logits = adapter.forward(
                        prepared[0], patches={clean_all: full_cache[donor_all]}
                    )
                    full_difference = float((full_logits - logits[1]).abs().max().item())
                    run.record_metadata(
                        "whole_state_positive_control",
                        {
                            "layer": p.layers[0],
                            "max_abs_logit_difference_from_donor": full_difference,
                            "answer": restored_donor["answer_text"],
                        },
                    )
                    if full_difference > 0.01:
                        raise RuntimeError("完整状态移植未恢复 donor，不能解释多位置干预")
                correct = all(
                    pred["answer_text"].strip().lower() == item.answer
                    for pred, item in zip(base, (pair.recipient, pair.donor), strict=True)
                )
                info = {
                    "pair_id": pair.pair_id,
                    "family_id": pair.family_id,
                    "split": pair.split,
                    "setting": pair.recipient.setting,
                    "relation": pair.recipient.relation,
                    "contrast": pair.contrast,
                    "labels": {
                        "clean": pair.clean_label,
                        "address": pair.address_label,
                        "attribute": pair.attribute_label,
                    },
                    "baseline_pair_correct": correct,
                }
                save(
                    stream,
                    {
                        **info,
                        "phase": "baseline",
                        "recipient": base[0],
                        "donor": base[1],
                        "recipient_tokens": prepared[0].token_record,
                        "donor_tokens": prepared[1].token_record,
                    },
                )
                for position in p.positions:
                    for layer in p.layers:
                        clean_site = Site(layer, "residual", prepared[0].roles[position])
                        donor_site = Site(layer, "residual", prepared[1].roles[position])
                        if index == 0:
                            noop, _, noop_logits = adapter.forward(
                                prepared[0], patches={clean_site: caches[0][clean_site]}
                            )
                            max_diff = float((noop_logits - logits[0]).abs().max().item())
                            save(
                                stream,
                                {
                                    **info,
                                    "phase": "noop",
                                    "position": position,
                                    "layer": layer,
                                    "max_abs_logit_diff": max_diff,
                                    "answer_unchanged": noop["answer_text"]
                                    == base[0]["answer_text"],
                                },
                            )
                            if max_diff > 0.01:
                                raise RuntimeError("自身替换改变原生 logits，停止解释干预结果")
                        pred, _, _ = adapter.forward(
                            prepared[0], patches={clean_site: caches[1][donor_site]}
                        )
                        save(
                            stream,
                            {
                                **info,
                                "phase": "patch",
                                "position": position,
                                "layer": layer,
                                "recipient_physical_indices": list(clean_site.positions),
                                "donor_physical_indices": list(donor_site.positions),
                                **scored(pred, pair),
                            },
                        )
                if index == 0:
                    restored, _, restored_logits = adapter.forward(prepared[0])
                    difference = float((restored_logits - logits[0]).abs().max().item())
                    save(
                        stream,
                        {
                            **info,
                            "phase": "restoration",
                            "max_abs_logit_diff": difference,
                            "answer_unchanged": restored["answer_text"] == base[0]["answer_text"],
                        },
                    )
                    if difference > 0.01:
                        raise RuntimeError("干预后模型状态未恢复")
                run.logger.info(
                    "%s 对 %s/%s，%s；双端基线正确=%s",
                    config.document.experiment.id,
                    index + 1,
                    len(active),
                    pair.recipient.setting,
                    correct,
                )
        run.artifacts.register(
            "patching/predictions.jsonl", description="逐对基线、真实概率、物理端口和恢复检查"
        )
        with run.time_phase("analysis"):
            result = summarize(
                rows, seed=config.document.run.seed, replicates=p.bootstrap_replicates
            )
            run.artifacts.write_json(
                "patching/summary.json", result, description="全部及基线正确子集的逐层曲线"
            )
        run.record_metadata(
            "execution",
            {
                "native_forward_calls": adapter.forward_calls,
                "additional_generate_calls": 1,
                "evaluated_splits": p.run_splits,
                "n_pairs": len(active),
            },
        )
        run.record_metadata("gpu_memory", adapter.peak_memory())
    return run.run_dir
