"""执行 E01 的三形状工程冒烟：真实推理、图像对照和逐样本记录，不代替正式机制实验。"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image
from pydantic import Field, field_validator

from reladdr.runtime import RunContext, load_config, resolve_path
from reladdr.runtime.config import StrictModel
from reladdr.tasks.synthetic import (
    GENERATOR_VERSION,
    PALETTE,
    make_scenes,
    questions_for,
    render_scene,
)


class Parameters(StrictModel):
    model_path: Path
    model_revision: str = Field(min_length=1)
    families: int = Field(default=1, ge=1, le=100)
    max_new_tokens: int = Field(default=8, ge=1, le=32)
    image_conditions: list[str] = Field(default_factory=lambda: ["original", "blank", "mismatch"])

    @field_validator("model_path", mode="before")
    @classmethod
    def path_from_string(cls, value):
        return Path(value) if isinstance(value, str) else value

    @field_validator("image_conditions")
    @classmethod
    def valid_conditions(cls, value):
        if (
            not value
            or len(value) != len(set(value))
            or set(value) - {"original", "blank", "mismatch"}
        ):
            raise ValueError("图像条件须为不重复的 original/blank/mismatch 非空列表")
        return value


def score_prediction(prediction: dict, answer: str) -> dict:
    match = re.fullmatch(r"([a-z]+)[.!]?", prediction["answer_text"].strip().lower())
    parsed = match.group(1) if match and match.group(1) in PALETTE else None
    scores = prediction["candidate_log_probs"]
    foil = max((color for color in PALETTE if color != answer), key=scores.__getitem__)
    return {
        "parsed_answer": parsed,
        "format_valid": parsed is not None,
        "matches_original_answer": parsed == answer,
        "candidate_prediction": max(scores, key=scores.__getitem__),
        "strongest_wrong_color": foil,
        "correct_foil_margin": scores[answer] - scores[foil],
    }


def run_baseline(config, *, adapter_factory=None) -> Path:
    if adapter_factory is None:
        from reladdr.models.qwen2_vl import Qwen2VLAdapter

        adapter_factory = Qwen2VLAdapter
    parameters = config.parameters
    model_path = resolve_path(parameters.model_path, config.source.parent)
    with RunContext(config) as run:
        run.record_metadata(
            "protocol",
            {
                "stage": "engineering_smoke",
                "generator": GENERATOR_VERSION,
                "split": "debug",
                "model_revision": parameters.model_revision,
                "prompt_protocol": "reference_first_chat_color_v1",
                "answer_protocol": "one_color_word; lower/title next-token logsumexp",
                "color_palette": PALETTE,
            },
        )
        scenes = make_scenes(config.document.run.seed, parameters.families)
        images = {}
        for scene in scenes:
            image = render_scene(scene)
            images[scene.scene_id] = image
            name = f"dataset/images/{scene.scene_id}.png"
            image.save(run.artifacts.path_for(name))
            run.artifacts.register(name, description="精确标注的三形状图像")
        run.artifacts.write_jsonl(
            "dataset/scenes.jsonl",
            (scene.record() for scene in scenes),
            description="对象与场景族真值",
        )
        run.artifacts.write_jsonl(
            "dataset/questions.jsonl",
            (
                {"scene_id": scene.scene_id, "family_id": scene.family_id, **question.record()}
                for scene in scenes
                for question in questions_for(scene)
            ),
            description="完整调试问题及真值",
        )
        with run.time_phase("model_load"):
            adapter = adapter_factory(model_path, max_new_tokens=parameters.max_new_tokens)
        run.record_metadata("model", adapter.metadata)
        summaries = defaultdict(
            lambda: {
                "n": 0,
                "n_labeled": 0,
                "correct": 0,
                "original_matches": 0,
                "valid": 0,
                "margin_sum": 0.0,
            }
        )

        def rows():
            for scene in scenes:
                image = images[scene.scene_id]
                # 错配图保持同一布局，但循环置换颜色；每个问题的正确颜色都会改变。
                mismatch_scene = type(scene)(
                    scene.scene_id,
                    scene.family_id,
                    scene.orientation,
                    tuple(
                        type(obj)(
                            obj.object_id, obj.shape, scene.objects[(index + 1) % 3].color, obj.bbox
                        )
                        for index, obj in enumerate(scene.objects)
                    ),
                    scene.size,
                )
                variants = {
                    "original": image,
                    "blank": Image.new("RGB", image.size, "white"),
                    "mismatch": render_scene(mismatch_scene),
                }
                for condition in parameters.image_conditions:
                    name = f"dataset/images/{scene.scene_id}-{condition}.png"
                    variants[condition].save(run.artifacts.path_for(name))
                    run.artifacts.register(name, description="模型实际接收的图像条件")
                for question in questions_for(scene):
                    for condition in parameters.image_conditions:
                        prediction = adapter.predict(
                            variants[condition],
                            question.prompt,
                            candidate_aliases={
                                color: [color, color.capitalize()] for color in PALETTE
                            },
                        )
                        scored = score_prediction(prediction, question.answer)
                        aggregate = summaries[f"{condition}.{question.kind}"]
                        aggregate["n"] += 1
                        # 错配图按实际呈现的颜色计正确率；空白图没有答案，不生成假准确率。
                        displayed_answer = None
                        if condition == "original":
                            displayed_answer = question.answer
                        elif condition == "mismatch":
                            displayed_answer = next(
                                obj.color
                                for obj in mismatch_scene.objects
                                if obj.object_id == question.target_id
                            )
                        correct = (
                            scored["parsed_answer"] == displayed_answer
                            if displayed_answer
                            else None
                        )
                        if correct is not None:
                            aggregate["n_labeled"] += 1
                            aggregate["correct"] += correct
                        aggregate["original_matches"] += scored["matches_original_answer"]
                        aggregate["valid"] += scored["format_valid"]
                        aggregate["margin_sum"] += scored["correct_foil_margin"]
                        run.logger.info(
                            "%s %s answer=%s expected=%s",
                            question.question_id,
                            condition,
                            prediction["answer_text"],
                            question.answer,
                        )
                        yield {
                            "scene_id": scene.scene_id,
                            "family_id": scene.family_id,
                            "condition": condition,
                            "displayed_answer": displayed_answer,
                            "correct": correct,
                            "image_path": (
                                f"artifacts/dataset/images/{scene.scene_id}-{condition}.png"
                            ),
                            **question.record(),
                            **prediction,
                            **scored,
                        }

        with run.time_phase("evaluation"):
            run.artifacts.write_jsonl(
                "baseline/predictions.jsonl", rows(), description="逐问真实输出与图像对照"
            )
        summary = {}
        for slice_name, values in summaries.items():
            summary[slice_name] = {
                "n": values["n"],
                "n_labeled": values["n_labeled"],
                "original_label_match_rate": values["original_matches"] / values["n"],
                "format_valid_rate": values["valid"] / values["n"],
                "mean_correct_foil_margin": values["margin_sum"] / values["n"],
            }
            if values["n_labeled"]:
                summary[slice_name]["accuracy"] = values["correct"] / values["n_labeled"]
            for metric, value in summary[slice_name].items():
                run.log_metric(f"{slice_name}.{metric}", value)
        run.artifacts.write_json(
            "baseline/summary.json",
            summary,
            description="按图像条件与任务切片的调试结果",
            sync=True,
        )
        memory = adapter.peak_memory()
        run.record_metadata("gpu_memory", memory)
        for name, value in memory.items():
            run.log_metric(f"gpu.{name}", value, unit="bytes")
    return run.run_dir


def main():
    parser = argparse.ArgumentParser(description="E01 工程冒烟，不执行机制干预")
    parser.add_argument("config", help="E01 私有 YAML 配置")
    args = parser.parse_args()
    config = load_config(args.config, parameter_model=Parameters)
    directory = run_baseline(config)
    print(json.dumps({"run_dir": str(directory)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
