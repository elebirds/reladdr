"""验证临时钩子的异常清理及张量切片替换；张量测试仅使用已有可选 Torch。"""

import pytest

from reladdr.interventions.hooks import Site, replace_rows, select_rows, temporary_hooks


class Handle:
    def __init__(self, events):
        self.events = events

    def remove(self):
        self.events.append("remove")


class Module:
    def __init__(self, events):
        self.events = events

    def register_forward_hook(self, callback):
        self.events.append("post")
        return Handle(self.events)

    def register_forward_pre_hook(self, callback):
        self.events.append("pre")
        return Handle(self.events)


def test_hooks_removed_after_exception_and_partial_registration():
    events = []
    module = Module(events)
    with (
        pytest.raises(RuntimeError),
        temporary_hooks([(module, "pre", None), (module, "post", None)]),
    ):
        raise RuntimeError("controlled")
    assert events == ["pre", "post", "remove", "remove"]
    events.clear()
    with (
        pytest.raises(ValueError),
        temporary_hooks([(module, "post", None), (module, "invalid", None)]),
    ):
        pass
    assert events == ["post", "remove"]


def test_head_and_position_validation():
    for args in [
        (-1, "residual", (0,)),
        (0, "residual", ()),
        (0, "residual", (0, 0)),
        (0, "unknown", (0,)),
        (0, "residual", (0,), 1),
    ]:
        with pytest.raises(ValueError):
            Site(*args)


def test_tensor_patch_preserves_other_rows_and_heads():
    torch = pytest.importorskip("torch")
    original = torch.arange(24).reshape(1, 3, 8).float()
    donor = torch.full((1, 4), -3.0)
    site = Site(0, "heads", (1,), head=1)
    changed = replace_rows(original, site, donor, 4)
    assert torch.equal(select_rows(changed, site, 4), donor)
    assert torch.equal(changed[0, 0], original[0, 0])
    assert torch.equal(changed[0, 1, :4], original[0, 1, :4])
    assert torch.equal(original, torch.arange(24).reshape(1, 3, 8).float())
    with pytest.raises(ValueError):
        replace_rows(original, site, torch.zeros(4), 4)
