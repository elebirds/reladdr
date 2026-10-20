"""兼容旧 e02 入口；共享实际模块对象，保持导入与模块级替换行为。"""

import sys

from reladdr.experiments.b02_binding_replication import runner as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    # 不复制函数到薄包装层，避免旧入口的 monkeypatch 与实际执行模块脱节。
    sys.modules[__name__] = _implementation
