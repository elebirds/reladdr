"""验证组件冻结的对照只改变自然语言条件，不改变图像或答案真值。"""

from reladdr.experiments.e03_freeze import condition_pairs


def test_condition_changes_keep_same_image_and_change_target():
    rows = condition_pairs(19, {"discovery": 2, "validation": 1, "test": 1})
    assert {row[1] for row in rows} == {"relation_only", "reference_only"}
    for _key, contrast, old, new in rows:
        assert old.objects == new.objects
        assert old.answer != new.answer
        assert new.answer == new.objects[new.target_index].color
        assert old.prompt != new.prompt
        if contrast == "relation_only":
            assert old.reference == new.reference
        else:
            assert old.relation == new.relation


def test_freeze_defaults_declare_both_actual_task_settings():
    import pytest
    from pydantic import ValidationError

    from reladdr.experiments.e03_freeze import Parameters

    parameters = Parameters(model_path="test-only", model_revision="test-only")
    assert parameters.settings == ["squares_horizontal", "squares_grid2"]
    with pytest.raises(ValidationError):
        Parameters(model_path="test-only", model_revision="test-only", settings=["shapes_vertical"])
