"""记录代码、依赖与主机来源，不导入模型栈、不收集环境变量全集。"""

import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path

from reladdr.runtime.security import redact_text


def _git(directory: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(directory), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.rstrip("\n")


def capture_provenance(code_root: Path | None = None) -> tuple[dict, str, dict[str, str]]:
    """返回来源、相关代码差异及未跟踪源码；后两者只保存在本地。"""
    packages = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"]
        if name:
            # 按 Python 搜索路径保留首个可见分发；继承环境中的旧包不能覆盖项目环境版本。
            packages.setdefault(name, distribution.version)
    metadata = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "entrypoint": Path(sys.argv[0]).name,
        "packages": dict(sorted(packages.items())),
        "git": {"available": False},
    }
    module_file = Path(__file__).resolve()
    directory = code_root or module_file.parents[3]
    # wheel 位于另一个项目的虚拟环境里时，不能把那个项目的 Git 误记为包的来源。
    if code_root is None and module_file != directory / "src/reladdr/runtime/provenance.py":
        metadata["git"]["reason"] = "当前为已安装分发包，未显式关联源码 checkout"
        return metadata, "", {}
    try:
        root = Path(_git(directory, "rev-parse", "--show-toplevel"))
        commit = _git(root, "rev-parse", "HEAD")
        status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
        # 禁止外部 diff/textconv，避免读取版本时触发仓库配置中的程序。
        diff = _git(
            root,
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "HEAD",
            "--",
            "src",
            "pyproject.toml",
            "uv.lock",
            ".python-version",
        )
        untracked = _git(root, "ls-files", "--others", "--exclude-standard", "--", "src")
    except (FileNotFoundError, subprocess.SubprocessError):
        metadata["git"]["reason"] = "未获得 Git 来源；正式结果应从已提交的 checkout 运行"
        return metadata, "", {}
    snapshots = {}
    omitted = []
    for relative in untracked.splitlines():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or path.suffix not in {".py", ".toml", ".yaml", ".yml"}:
            omitted.append(relative)
            continue
        try:
            snapshots[relative] = redact_text(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError):
            omitted.append(relative)
    metadata["git"] = {
        "available": True,
        "root": str(root),
        "commit": commit,
        "dirty": bool(status),
        "changes": status.splitlines(),
        "untracked_source_files": untracked.splitlines(),
        "unsaved_untracked_files": omitted,
    }
    return metadata, redact_text(diff), snapshots
