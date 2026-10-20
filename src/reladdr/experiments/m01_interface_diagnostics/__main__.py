"""提供 M01 一维与二维接口诊断的命令行分派，不自动加载模型。"""

import argparse

from reladdr.experiments.m01_interface_diagnostics import grid, linear
from reladdr.runtime import load_config


def main():
    parser = argparse.ArgumentParser(description="M01_Interface_Diagnostics：已实现的接口诊断")
    commands = parser.add_subparsers(dest="variant", required=True)
    for name, module, runner, description in (
        ("linear", linear, linear.run_e03, "一维同指令参照移位"),
        ("grid", grid, grid.run_grid, "二维参照移位，排除参照色重叠"),
    ):
        command = commands.add_parser(name, help=description, description=description)
        command.add_argument("config", help="声明对应既有协议的私有 YAML 配置")
        command.add_argument("--prepare-only", action="store_true", help="仅准备数据，不加载模型")
        command.set_defaults(parameter_model=module.Parameters, runner=runner)
    args = parser.parse_args()
    config = load_config(args.config, parameter_model=args.parameter_model)
    print(args.runner(config, prepare_only=args.prepare_only))


if __name__ == "__main__":
    main()
