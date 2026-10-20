"""验证分组中介覆盖实际后缀，并且充分/必要替换没有污染原始状态。"""

from types import SimpleNamespace

import pytest

from reladdr.experiments.e03_path import mix_group, token_groups


def test_groups_are_disjoint_and_cover_suffix():
    prepared = SimpleNamespace(
        roles={"text_suffix": tuple(range(10, 23)), "reference": (15, 16), "relation": (13,)},
        token_record={"user_message_end_index": 19},
    )
    groups = token_groups(prepared)
    combined = [i for positions in groups.values() for i in positions]
    assert len(set(combined)) == len(combined) == 13
    assert set(combined) == set(range(10, 23))
    assert groups["question_tail"] == (17, 18)
    assert groups["assistant"] == (19, 20, 21, 22)


def test_group_mixing_preserves_sources_and_other_rows():
    torch = pytest.importorskip("torch")
    clean = torch.zeros(4, 3)
    donor = torch.ones(4, 3)
    sufficient = mix_group(clean, donor, [1], mode="group_only")
    necessary = mix_group(clean, donor, [1], mode="full_except_group")
    assert torch.equal(sufficient.sum(dim=1), torch.tensor([0.0, 3.0, 0.0, 0.0]))
    assert torch.equal(necessary.sum(dim=1), torch.tensor([3.0, 0.0, 3.0, 3.0]))
    assert clean.sum().item() == 0 and donor.sum().item() == 12
