"""描述明确的原生张量端口，并保证临时前向钩子在成功或异常后都被移除。"""

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Site:
    layer: int
    component: Literal["residual", "attention", "mlp", "heads", "query", "key", "value"]
    positions: tuple[int, ...]
    head: int | None = None

    def __post_init__(self):
        if self.layer < 0 or not self.positions or min(self.positions) < 0:
            raise ValueError("层和物理 token 索引必须有效且非空")
        if len(self.positions) != len(set(self.positions)):
            raise ValueError("同一端口不能重复选择 token")
        if self.component not in {"residual", "attention", "mlp", "heads", "query", "key", "value"}:
            raise ValueError("未知原生组件")
        if self.head is not None and (self.head < 0 or self.component != "heads"):
            raise ValueError("仅投影前的 heads 端口接受非负 query-head 编号")


@contextmanager
def temporary_hooks(registrations):
    """注册成功多少就清理多少，不重置模型、不留下影响后续样本的钩子。"""
    with ExitStack() as stack:
        for module, kind, callback in registrations:
            if kind == "pre":
                handle = module.register_forward_pre_hook(callback)
            elif kind == "post":
                handle = module.register_forward_hook(callback)
            else:
                raise ValueError("未知钩子类型")
            stack.callback(handle.remove)
        yield


def select_rows(tensor, site: Site, head_dim: int):
    if tensor.ndim != 3 or tensor.shape[0] != 1 or max(site.positions) >= tensor.shape[1]:
        raise ValueError("干预要求 batch=1，且物理索引落在当前序列内")
    columns = slice(None)
    if site.head is not None:
        if (site.head + 1) * head_dim > tensor.shape[-1]:
            raise ValueError("query-head 超出原生张量范围")
        columns = slice(site.head * head_dim, (site.head + 1) * head_dim)
    return tensor[0, list(site.positions), columns]


def replace_rows(tensor, site: Site, values, head_dim: int):
    selected = select_rows(tensor, site, head_dim)
    if selected.shape != values.shape:
        raise ValueError("donor 与 recipient 的端口形状不匹配，不自动广播")
    changed = tensor.clone()
    columns = slice(None)
    if site.head is not None:
        columns = slice(site.head * head_dim, (site.head + 1) * head_dim)
    # 不原地修改模型输出或 donor；其他 token、其他头和残差路径保持原值。
    changed[0, list(site.positions), columns] = values.to(device=tensor.device, dtype=tensor.dtype)
    return changed
