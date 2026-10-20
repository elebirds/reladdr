"""提供 M02 分组中介与条件写入冻结的命令行分派，不创建计划阶段空入口。"""

import argparse

from reladdr.experiments.m02_path_diagnostics import freeze, mediation
from reladdr.runtime import load_config


def main():
    parser = argparse.ArgumentParser(description="M02_Path_Diagnostics：已实现的路径诊断")
    commands = parser.add_subparsers(dest="variant", required=True)
    for name, module, runner, description in (
        ("mediation", mediation, mediation.run_path, "完整后缀的分组中介"),
        ("freeze", freeze, freeze.run_freeze, "旧条件文本组件写入冻结"),
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
