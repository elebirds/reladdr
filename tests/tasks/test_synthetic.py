"""核对三形状生成、渲染、分组和逐问关系真值，不加载模型。"""

import random

import pytest

from reladdr.tasks.synthetic import PALETTE, make_scenes, questions_for, render_scene


def test_generation_is_deterministic_and_does_not_change_global_rng():
    before = random.getstate()
    scenes = make_scenes(19, 3)
    assert scenes == make_scenes(19, 3)
    assert before == random.getstate()
    assert len(scenes) == 6
    assert len({scene.family_id for scene in scenes}) == 3
    for horizontal, vertical in zip(scenes[::2], scenes[1::2], strict=True):
        assert horizontal.family_id == vertical.family_id
        assert [obj.color for obj in horizontal.objects] == [obj.color for obj in vertical.objects]
        assert [obj.shape for obj in horizontal.objects] == [obj.shape for obj in vertical.objects]


def test_question_labels_follow_unique_objects_and_adjacent_positions():
    ids = []
    for scene in make_scenes(7, 2):
        questions = questions_for(scene)
        assert len(questions) == 10
        assert len({obj.shape for obj in scene.objects}) == 3
        assert len({obj.color for obj in scene.objects}) == 3
        objects = {obj.object_id: obj for obj in scene.objects}
        for question in questions:
            ids.append(question.question_id)
            assert objects[question.target_id].color == question.answer
            assert question.foil != question.answer
            assert question.foil in {obj.color for obj in scene.objects}
            if question.kind == "relation":
                reference = objects[question.reference_id]
                target = objects[question.target_id]
                axis = 0 if scene.orientation == "horizontal" else 1
                delta = target.bbox[axis] - reference.bbox[axis]
                assert abs(delta) == 140
                assert (delta < 0) == (question.relation in {"left", "above"})
    assert len(ids) == len(set(ids))


def test_render_matches_object_geometry_and_color():
    for scene in make_scenes(7, 2):
        image = render_scene(scene)
        assert image.size == (448, 448)
        assert image.mode == "RGB"
        assert image.getpixel((0, 0)) == (255, 255, 255)
        for obj in scene.objects:
            x0, y0, x1, y1 = obj.bbox
            assert image.getpixel(((x0 + x1) // 2, (y0 + y1) // 2)) == PALETTE[obj.color]


@pytest.mark.parametrize("families", [0, -1, True, 1.5])
def test_invalid_family_count_is_not_silently_accepted(families):
    with pytest.raises(ValueError):
        make_scenes(7, families)
