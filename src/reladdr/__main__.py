"""允许通过 python -m reladdr 调用公共运行工具。"""

from reladdr.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
