"""执行 B01 正式行为基线，兼容 E01 的分组、视觉对照、模板和统计协议。"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw
from pydantic import Field, StrictInt, field_validator, model_validator

from reladdr.evaluation.clustered import clustered_interval
from reladdr.experiments.e01 import score_prediction
from reladdr.runtime import RunContext, load_config, resolve_path
from reladdr.runtime.config import StrictModel
from reladdr.tasks.relational import (
    CONDITIONS,
    GENERATOR_VERSION,
    LAYOUTS,
    SPLITS,
    TEMPLATES,
    combination_partitions,
    dataset_counts,
    make_grouped_scenes,
    questions_for,
    visual_condition,
)
from reladdr.tasks.synthetic import PALETTE, render_objects

PROTOCOL_VERSION = "e01-v1"
EXPERIMENT_ID = "B01_Behavioral_Baseline"


class Parameters(StrictModel):
    model_path: Path
    model_revision: str = Field(min_length=1)
    model_source_manifest: Path | None = None
    system_prompt: str = Field(
        default="You are a helpful assistant. You respond in one token.", min_length=1
    )
    family_counts: dict[str, StrictInt] = Field(
        default_factory=lambda: {"discovery": 8, "validation": 4, "test": 4}
    )
    run_splits: list[str] = Field(default_factory=lambda: ["discovery"])
    layouts: list[str] = Field(default_factory=lambda: list(LAYOUTS))
    templates: list[str] = Field(default_factory=lambda: list(TEMPLATES))
    conditions: list[str] = Field(default_factory=lambda: list(CONDITIONS))
    combination_seed: int = Field(default=20261020, ge=0)
    max_new_tokens: int = Field(default=8, ge=1, le=32)
    bootstrap_replicates: int = Field(default=1000, ge=100)

    @field_validator("model_path", "model_source_manifest", mode="before")
    @classmethod
    def path_from_string(cls, value):
        return Path(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_protocol(self):
        for name, allowed in (
            ("run_splits", SPLITS),
            ("layouts", LAYOUTS),
            ("templates", TEMPLATES),
            ("conditions", CONDITIONS),
        ):
            values = getattr(self, name)
            if not values or len(values) != len(set(values)) or set(values) - set(allowed):
                raise ValueError(f"{name} 必须是已知值组成的不重复非空列表")
        if "original" not in self.conditions:
            raise ValueError("成对比较必须包含 original")
        if set(self.family_counts) != set(SPLITS) or any(
            type(n) is not int or n < 1 for n in self.family_counts.values()
        ):
            raise ValueError("三个 split 必须各有正整数场景族数量")
        if sum(self.family_counts.values()) > 120:
            raise ValueError("本版生成协议总场景族不能超过 120")
        return self


def _preview(run, scenes, parameters):
    selected = [scene for scene in scenes if scene.family_id == scenes[0].family_id]
    tile = 224
    sheet = Image.new("RGB", (3 * tile, len(selected) * (tile + 36)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, scene in enumerate(selected):
        question = next(
            q
            for q in questions_for(
                scene, templates=parameters.templates, combination_seed=parameters.combination_seed
            )
            if q.kind == "relation"
        )
        for column, condition in enumerate(("original", "target_recolor", "position_swap")):
            image, details = visual_condition(scene, question, condition)
            x, y = column * tile, row * (tile + 36)
            draw.text((x + 4, y + 3), f"{scene.layout} / {condition}", fill="black")
            draw.text(
                (x + 4, y + 17),
                f"{question.relation} -> {details['displayed_answer']}",
                fill="black",
            )
            sheet.paste(image.resize((tile, tile)), (x, y + 36))
    name = "dataset/preview.png"
    sheet.save(run.artifacts.path_for(name))
    run.artifacts.register(name, description="首个场景族的原图、换色和位置交换实际渲染")


def _interval(rows, field, parameters, seed):
    totals = defaultdict(lambda: [0.0, 0])
    for row in rows:
        value = row.get(field)
        if value is not None:
            totals[row["family_id"]][0] += float(value)
            totals[row["family_id"]][1] += 1
    return clustered_interval(totals, replicates=parameters.bootstrap_replicates, seed=seed)


def _slices(rows, fields, parameters, seed):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in fields)].append(row)
    result = []
    for key, selected in sorted(grouped.items()):
        result.append(
            {
                **dict(zip(fields, key, strict=True)),
                "n": len(selected),
                "accuracy": _interval(selected, "correct", parameters, seed),
                "candidate_accuracy": _interval(selected, "candidate_correct", parameters, seed),
                "format_valid_rate": sum(r["format_valid"] for r in selected) / len(selected),
                "original_label_match_rate": sum(r["matches_original_answer"] for r in selected)
                / len(selected),
                "mean_original_margin": _interval(
                    selected, "original_answer_margin", parameters, seed
                ),
            }
        )
    return result


def _pair_summary(rows, parameters, seed):
    originals = {row["question_id"]: row for row in rows if row["condition"] == "original"}
    pairs = []

    def add(left, right, contrast):
        if left["correct"] is None or right["correct"] is None:
            return
        changed = left["displayed_answer"] != right["displayed_answer"]
        pairs.append(
            {
                "family_id": left["family_id"],
                "split": left["split"],
                "layout": left["layout"],
                "template": left["template"],
                "contrast": contrast,
                "expected_answer_changed": changed,
                "both_correct": left["correct"] and right["correct"],
                "answer_changed": left["parsed_answer"] != right["parsed_answer"],
                "margin_change": right["original_answer_margin"] - left["original_answer_margin"]
                if contrast in CONDITIONS
                else None,
            }
        )

    for row in rows:
        if row["condition"] not in {"original", "blank"}:
            add(originals[row["question_id"]], row, row["condition"])
    relations = [row for row in originals.values() if row["kind"] == "relation"]
    groups = defaultdict(list)
    for row in relations:
        groups[row["scene_id"], row["template"], row["relation"]].append(row)
    # 合法同关系问题按参照槽位排序，固定比较相邻参照；不按模型对错挑配对。
    for group in groups.values():
        ordered = sorted(group, key=lambda row: row["reference_id"])
        for left, right in zip(ordered, ordered[1:], strict=False):
            add(left, right, "reference_change")
    by_reference = {
        (row["scene_id"], row["template"], row["reference_id"], row["relation"]): row
        for row in relations
    }
    for row in relations:
        opposite = {"left": "right", "above": "below"}.get(row["relation"])
        if opposite and (
            other := by_reference.get(
                (row["scene_id"], row["template"], row["reference_id"], opposite)
            )
        ):
            add(row, other, "relation_change")
    grouped = defaultdict(list)
    for pair in pairs:
        grouped[pair["split"], pair["layout"], pair["template"], pair["contrast"]].append(pair)
    summary = []
    for (split, layout, template, contrast), selected in sorted(grouped.items()):
        summary.append(
            {
                "split": split,
                "layout": layout,
                "template": template,
                "contrast": contrast,
                "n": len(selected),
                "expected_answer_changed_n": sum(p["expected_answer_changed"] for p in selected),
                "both_correct": _interval(selected, "both_correct", parameters, seed),
                "answer_changed_rate": _interval(selected, "answer_changed", parameters, seed),
                "mean_fixed_margin_change": _interval(selected, "margin_change", parameters, seed),
            }
        )
    return pairs, summary


def summarize(rows, parameters, seed):
    pairs, paired_summary = _pair_summary(rows, parameters, seed)
    originals = [row for row in rows if row["condition"] == "original"]
    controls = {
        (row["scene_id"], row["kind"], row["target_id"]): row
        for row in originals
        if row["kind"] != "relation"
    }
    patterns = Counter()
    for row in originals:
        if row["kind"] != "relation":
            continue
        reference = controls[row["scene_id"], "recognition", row["reference_id"]]
        target = controls[row["scene_id"], "recognition", row["target_id"]]
        absolute = controls[row["scene_id"], "absolute", row["target_id"]]
        patterns[
            row["layout"],
            row["template"],
            bool(reference["correct"]),
            bool(target["correct"]),
            bool(absolute["correct"]),
            bool(row["correct"]),
        ] += 1
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "n_predictions": len(rows),
        "n_families_evaluated": len({row["family_id"] for row in rows}),
        "evaluated_splits": sorted({row["split"] for row in rows}),
        "resampling_unit": "scene_family",
        "bootstrap_replicates": parameters.bootstrap_replicates,
        "slices": _slices(
            rows, ("split", "layout", "condition", "kind", "template"), parameters, seed
        ),
        "relation_slices": _slices(
            [r for r in rows if r["kind"] == "relation"],
            ("split", "condition", "template", "relation"),
            parameters,
            seed,
        ),
        "combination_slices": _slices(
            [r for r in rows if r["kind"] == "relation"],
            ("split", "condition", "combination_partition"),
            parameters,
            seed,
        ),
        "paired": paired_summary,
        "diagnostic_patterns": [
            {
                "layout": layout,
                "template": template,
                "reference_crop_correct": ref,
                "target_crop_correct": target,
                "absolute_correct": absolute,
                "relation_correct": relation,
                "n": n,
            }
            for (layout, template, ref, target, absolute, relation), n in sorted(patterns.items())
        ],
        "diagnostic_interpretation": "行为对照，不以 crop/位置读出对错直接归因内部计算",
    }
    return result, pairs


def _report(summary):
    lines = [
        "# E01 分组基线结果",
        "",
        f"实际评估 {summary['n_predictions']} 条输入、"
        f"{summary['n_families_evaluated']} 个场景族；split："
        f"{', '.join(summary['evaluated_splits'])}。",
        "",
        "识别为 oracle 单对象裁剪上限，与整图位置读取、关系寻址分别报告。",
        "区间按场景族重采样；空白图无准确率。该批数据不建立原生机制结论。",
        "",
        "| 布局 | 任务 | 模板 | n | 原图准确率 | 场景族 95% 区间 |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in summary["slices"]:
        if row["condition"] != "original":
            continue
        accuracy = row["accuracy"]
        ci = accuracy["ci95"]
        interval = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "场景族不足 2 个"
        lines.append(
            f"| {row['layout']} | {row['kind']} | {row['template']} | "
            f"{accuracy['n']} | {accuracy['estimate']:.3f} | {interval} |"
        )
    return "\n".join(lines) + "\n"


def run_formal(config, *, adapter_factory=None, prepare_only=False) -> Path:
    parameters = config.parameters
    if config.document.experiment.id not in {"E01", EXPERIMENT_ID} or (
        config.document.experiment.protocol_version != PROTOCOL_VERSION
    ):
        raise ValueError(
            f"正式入口要求 experiment.id={EXPERIMENT_ID} 或 E01、protocol_version=e01-v1"
        )
    scenes = make_grouped_scenes(
        config.document.run.seed, parameters.family_counts, parameters.layouts
    )
    active = [scene for scene in scenes if scene.split in parameters.run_splits]
    with RunContext(config) as run:
        run.record_metadata(
            "protocol",
            {
                "stage": "P1",
                "experiments": ["E01.1", "E01.2"],
                "generator": GENERATOR_VERSION,
                "evaluation_executed": not prepare_only,
                "run_splits": parameters.run_splits,
                "recognition": "oracle_crop_color_upper_bound",
                "relation_semantics": "occupied_adjacent_grid_cell",
                "answer_protocol": "one_color_word; lower/title logsumexp",
            },
        )
        run.artifacts.write_jsonl(
            "dataset/scenes.jsonl",
            (scene.record() for scene in scenes),
            description="全数据集场景族、固定划分与对象真值",
        )
        run.artifacts.write_jsonl(
            "dataset/questions.jsonl",
            (
                {
                    "scene_id": scene.scene_id,
                    "family_id": scene.family_id,
                    "split": scene.split,
                    "layout": scene.layout,
                    **question.record(),
                }
                for scene in scenes
                for question in questions_for(
                    scene,
                    templates=parameters.templates,
                    combination_seed=parameters.combination_seed,
                )
            ),
            description="按场景族和参照位置—关系组合划分的问题真值",
        )
        counts = dataset_counts(
            scenes, templates=parameters.templates, combination_seed=parameters.combination_seed
        )
        planned = sum(row["n"] for row in counts if row["split"] in parameters.run_splits)
        run.artifacts.write_json(
            "dataset/manifest.json",
            {
                "family_counts": parameters.family_counts,
                "counts": counts,
                "planned_model_calls": planned * len(parameters.conditions),
                "evaluated_splits": [] if prepare_only else parameters.run_splits,
                "combination_seed": parameters.combination_seed,
                "combination_partitions": {
                    layout: [
                        {"row": r, "column": c, "relation": relation, "partition": partition}
                        for (r, c, relation), partition in combination_partitions(
                            layout, parameters.combination_seed
                        ).items()
                    ]
                    for layout in parameters.layouts
                },
            },
            description="固定数据与组合留出清单；未运行的 split 不产生模型结果",
        )
        for scene in scenes:
            name = f"dataset/scenes/{scene.scene_id}.png"
            render_objects(scene.objects, scene.size).save(run.artifacts.path_for(name))
            run.artifacts.register(name, description="场景完整原图")
        _preview(run, active, parameters)
        if prepare_only:
            run.logger.info(
                "数据准备完成，计划模型调用 %s；本次未加载模型",
                planned * len(parameters.conditions),
            )
            return run.run_dir
        if adapter_factory is None:
            if parameters.model_revision.startswith("unverified"):
                raise ValueError("正式 GPU 运行必须提供固定模型 revision")
            from reladdr.models.qwen2_vl import Qwen2VLAdapter

            adapter_factory = Qwen2VLAdapter
        if parameters.model_source_manifest is not None:
            source_path = resolve_path(parameters.model_source_manifest, config.source.parent)
            run.artifacts.write_json(
                "provenance/model_source_files.json",
                json.loads(source_path.read_text(encoding="utf-8")),
                description="checkpoint 与 processor 的既有逐文件版本清单",
            )
        with run.time_phase("model_load"):
            adapter = adapter_factory(
                resolve_path(parameters.model_path, config.source.parent),
                max_new_tokens=parameters.max_new_tokens,
                system_prompt=parameters.system_prompt,
            )
        run.record_metadata(
            "model", {**adapter.metadata, "declared_revision": parameters.model_revision}
        )
        analysis_rows = []
        prediction_name = "baseline/predictions.jsonl"
        prediction_path = run.artifacts.path_for(prediction_name)
        saved_images = set()
        with run.time_phase("evaluation"):
            # 每条输出即时落盘；故障时保留已经执行的昂贵模型结果，不自动续跑或覆盖。
            with prediction_path.open("x", encoding="utf-8") as stream:
                for scene in active:
                    questions = questions_for(
                        scene,
                        templates=parameters.templates,
                        combination_seed=parameters.combination_seed,
                    )
                    for question in questions:
                        for condition in parameters.conditions:
                            image, details = visual_condition(scene, question, condition)
                            image_name = f"dataset/inputs/{question.pair_id}-{condition}.png"
                            if image_name not in saved_images:
                                image.save(run.artifacts.path_for(image_name))
                                run.artifacts.register(
                                    image_name, description="模型实际接收的输入图像"
                                )
                                saved_images.add(image_name)
                            prediction = adapter.predict(
                                image,
                                question.prompt,
                                candidate_aliases={
                                    color: [color, color.capitalize()] for color in PALETTE
                                },
                            )
                            scored = score_prediction(prediction, question.answer, question.foil)
                            answer = details["displayed_answer"]
                            row = {
                                "scene_id": scene.scene_id,
                                "family_id": scene.family_id,
                                "split": scene.split,
                                "layout": scene.layout,
                                "condition": condition,
                                "image_path": f"artifacts/{image_name}",
                                **question.record(),
                                **details,
                                **prediction,
                                **scored,
                                "correct": scored["parsed_answer"] == answer if answer else None,
                                "candidate_correct": scored["candidate_prediction"] == answer
                                if answer
                                else None,
                                "answer_compatible_object_ids": [
                                    obj["object_id"]
                                    for obj in details["displayed_scene"]["objects"]
                                    if obj["color"] == scored["parsed_answer"]
                                ]
                                if details["displayed_scene"]
                                and question.input_view == "full_scene"
                                else [],
                            }
                            stream.write(
                                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                            )
                            stream.flush()
                            analysis_rows.append(
                                {
                                    key: value
                                    for key, value in row.items()
                                    if key
                                    not in {
                                        "displayed_scene",
                                        "candidate_log_probs",
                                        "candidate_token_ids",
                                        "prompt",
                                    }
                                }
                            )
                            if len(analysis_rows) % 50 == 0:
                                run.log_metric(
                                    "evaluation.predictions_completed", len(analysis_rows)
                                )
                                run.logger.info(
                                    "E01 进度 %s/%s，%s %s",
                                    len(analysis_rows),
                                    planned * len(parameters.conditions),
                                    scene.layout,
                                    scene.family_id,
                                )
        run.artifacts.register(
            prediction_name, description="逐问原始输出、实际真值及固定 foil 指标"
        )
        with run.time_phase("analysis"):
            summary, pairs = summarize(analysis_rows, parameters, config.document.run.seed)
            run.artifacts.write_json(
                "baseline/summary.json",
                summary,
                description="场景族区间、模板/方向切片和成对一致性",
                sync=True,
            )
            run.artifacts.write_jsonl(
                "baseline/pairs.jsonl", pairs, description="预声明视觉、参照及关系比较的逐对结果"
            )
            run.artifacts.write_text(
                "baseline/report.md",
                _report(summary),
                description="本次 E01 可读结果与适用条件",
                sync=True,
            )
        memory = adapter.peak_memory()
        run.record_metadata("gpu_memory", memory)
        run.log_metric("evaluation.predictions_completed", len(analysis_rows))
        for key, value in memory.items():
            run.log_metric(f"gpu.{key}", value, unit="bytes")
    return run.run_dir


def main():
    parser = argparse.ArgumentParser(description="B01 正式分组行为基线（兼容 E01 协议）")
    parser.add_argument("config", help="B01 或兼容 E01 的私有配置")
    parser.add_argument("--prepare-only", action="store_true", help="仅生成并验证数据，不加载模型")
    args = parser.parse_args()
    config = load_config(args.config, parameter_model=Parameters)
    print(
        json.dumps(
            {"run_dir": str(run_formal(config, prepare_only=args.prepare_only))}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
