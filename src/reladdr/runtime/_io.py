"""提供有限数值 JSON、原子状态写入和 run 内路径约束。"""

import json
import os
import tempfile
from pathlib import Path

from reladdr.runtime.security import redact_text, reject_secret_fields


def clean_json(value: object) -> object:
    """保留 JSON 结构，并在序列化前处理已知认证值。"""
    reject_secret_fields(value)
    if isinstance(value, dict):
        return {key: clean_json(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(child) for child in value]
    return redact_text(value) if isinstance(value, str) else value


def json_text(value: object) -> str:
    # 禁止 NaN/Infinity，避免本地结果与远程工具对数值含义理解不一致。
    return json.dumps(clean_json(value), ensure_ascii=False, allow_nan=False, indent=2) + "\n"


def atomic_json(path: Path, value: object) -> None:
    """完整写入后原子替换，读者只会看到旧状态或新状态。"""
    text = json_text(value)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_json(path: Path) -> object:
    def invalid_constant(value: str) -> None:
        raise ValueError("运行记录包含非有限数值")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid_constant)


def run_path(root: Path, relative: str) -> Path:
    """拒绝绝对路径、父级跳转和指向 run 外部的符号链接。"""
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("产物必须使用 run 内的相对文件路径")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()) or resolved == root.resolve():
        raise ValueError("产物路径不能越出 run 目录")
    return resolved
