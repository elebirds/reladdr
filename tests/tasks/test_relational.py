"""检查正式 E01 的边界、稳定分组、组合留出以及视觉反事实中的实际参照和答案。"""

from dataclasses import replace

import pytest

from reladdr.tasks.relational import (
    CONDITIONS,
    DIRECTIONS,
    LAYOUTS,
    TEMPLATES,
    combination_partitions,
    displayed_target,
    make_grouped_scenes,
    questions_for,
    validate_scenes,
    visual_condition,
)
from reladdr.tasks.synthetic import PALETTE


def scenes(count=2):
    return make_grouped_scenes(19, {"discovery": count, "validation": 1, "test": 1}, list(LAYOUTS))


def questions(scene):
    return questions_for(scene, templates=list(TEMPLATES), combination_seed=20261020)


def test_stable_family_split_and_no_exact_scene_duplicates():
    first = scenes(1)
    assert first == scenes(1)
    expanded = {scene.scene_id: scene for scene in scenes(2)}
    assert all(expanded[scene.scene_id] == scene for scene in first)
    assert len(first) == 15
    by_family = {}
    for scene in first:
        assert by_family.setdefault(scene.family_id, scene.split) == scene.split
        assert len({(obj.shape, obj.color) for obj in scene.objects}) == len(scene.objects)
        assert len(scene.objects) == scene.rows * scene.columns
    with pytest.raises(ValueError, match="跨数据划分"):
        validate_scenes([first[0], replace(first[0], split="invalid")])


def test_combination_holdouts_are_axis_balanced_and_never_leak():
    for layout in ("grid3", "grid4"):
        partitions = combination_partitions(layout, 20261020)
        for relation in DIRECTIONS:
            assert {
                partition
                for (*_, direction), partition in partitions.items()
                if direction == relation
            } == {"discovery", "validation_only", "test_only"}
    for scene in scenes(1):
        allowed = {"shared", "discovery", "not_applicable", f"{scene.split}_only"}
        for question in questions(scene):
            assert question.combination_partition in allowed
            assert displayed_target(scene, question).color == question.answer
            assert question.foil != question.answer
            if question.kind == "relation":
                ref = scene.index(question.reference_id)
                row, column = divmod(ref, scene.columns)
                dr, dc = DIRECTIONS[question.relation]
                assert question.target_cell == (row + dr, column + dc)


def test_visual_controls_preserve_reference_and_have_recomputed_unique_truth():
    seen = set()
    for scene in scenes(1):
        for question in questions(scene):
            for condition in CONDITIONS:
                image, details = visual_condition(scene, question, condition)
                seen.add((scene.layout, condition))
                assert image.mode == "RGB"
                if condition == "blank":
                    assert details["displayed_answer"] is None
                    assert image.getextrema() == ((255, 255),) * 3
                    continue
                if condition in {"target_recolor", "position_swap"}:
                    assert details["displayed_answer"] != question.answer
                assert details["displayed_answer"] in PALETTE
                if question.kind == "relation":
                    source_ref = next(
                        obj for obj in scene.objects if obj.object_id == question.reference_id
                    )
                    actual_ref = next(
                        obj
                        for obj in details["displayed_scene"]["objects"]
                        if obj["object_id"] == question.reference_id
                    )
                    assert actual_ref["bbox"] == source_ref.bbox
                    assert actual_ref["shape"] == source_ref.shape
                    assert actual_ref["color"] == source_ref.color
    assert len(seen) == len(LAYOUTS) * len(CONDITIONS)


def test_reference_order_templates_have_opposite_token_word_order():
    scene = scenes(1)[0]
    selected = [q for q in questions(scene) if q.kind == "relation"]
    for question in selected:
        reference = f"{question.reference_color} {question.reference_shape}"
        if question.template == "relation_first":
            assert question.prompt.index(question.relation) < question.prompt.index(reference)
        else:
            assert question.prompt.index(reference) < question.prompt.index(question.relation)


@pytest.mark.parametrize(
    "counts",
    [
        {"discovery": 0, "validation": 1, "test": 1},
        {"discovery": 1, "validation": 1},
        {"discovery": True, "validation": 1, "test": 1},
        {"discovery": 73, "validation": 1, "test": 1},
    ],
)
def test_invalid_family_specs_fail_before_model_loading(counts):
    with pytest.raises(ValueError):
        make_grouped_scenes(19, counts, ["grid3"])
