"""管理实验私有产物：限定目录、禁止覆盖，并登记允许同步的文件。"""

import json
from collections.abc import Iterable
from pathlib import Path

from reladdr.runtime._io import atomic_json, clean_json, json_text, run_path, sync_content_digest
from reladdr.runtime.security import redact_text


class ArtifactStore:
    """每个 run 由一个写入者持有；同步只读取已登记的产物。"""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.directory = run_dir / "artifacts"
        self.directory.mkdir()
        self._entries: list[dict] = []
        self._closed = False
        self._save_manifest()

    def _save_manifest(self) -> None:
        atomic_json(
            self.run_dir / "manifest.json", {"schema_version": 1, "artifacts": self._entries}
        )

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("run 已结束，不能继续修改产物")

    def path_for(self, name: str) -> Path:
        """供绘图库等外部写入者获取新路径；写好后必须调用 register。"""
        self._check_open()
        path = run_path(self.directory, name)
        if path.exists():
            raise FileExistsError("产物已存在，不能覆盖")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def register(self, name: str, *, description: str, sync: bool = False) -> Path:
        self._check_open()
        if not isinstance(sync, bool):
            raise ValueError("产物同步标记必须是布尔值")
        path = run_path(self.directory, name)
        if not path.is_file():
            raise ValueError("只能登记已经写完的文件")
        relative = path.relative_to(self.run_dir).as_posix()
        if any(entry["path"] == relative for entry in self._entries):
            raise ValueError("产物已登记，不能覆盖登记信息")
        stat = path.stat()
        entry = {
            "path": relative,
            "description": redact_text(description),
            "sync": sync,
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        if sync:
            # Linux 实测同长度改写可能保留相同时间戳；白名单文件需要内容完整性依据。
            entry["content_sha256"] = sync_content_digest(path)
        self._entries.append(entry)
        try:
            self._save_manifest()
        except BaseException:
            self._entries.pop()
            raise
        return path

    def write_json(self, name: str, value: object, *, description: str, sync: bool = False) -> Path:
        text = json_text(value)
        return self.write_text(name, text, description=description, sync=sync)

    def write_text(self, name: str, text: str, *, description: str, sync: bool = False) -> Path:
        path = self.path_for(name)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(redact_text(text))
        return self.register(name, description=description, sync=sync)

    def write_jsonl(
        self, name: str, records: Iterable[object], *, description: str, sync: bool = False
    ) -> Path:
        """逐条写入，避免把全部逐样本结果常驻内存。"""
        path = self.path_for(name)
        with path.open("x", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(clean_json(record), ensure_ascii=False, allow_nan=False))
                stream.write("\n")
        # 迭代器或序列化失败时保留未完成文件，但不登记、不上传它。
        return self.register(name, description=description, sync=sync)

    def close(self) -> None:
        self._closed = True
