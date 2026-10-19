"""提供公共配置检查和本地运行记录查看，不伪装成科学实验执行器。"""

import argparse
import json
import sys
from contextlib import redirect_stdout

from reladdr.runtime import ConfigError, load_config, read_run


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reladdr", description="RelAddr 公共运行工具")
    groups = parser.add_subparsers(dest="group", required=True)
    config = groups.add_parser("config", help="配置检查")
    config_commands = config.add_subparsers(dest="command", required=True)
    check = config_commands.add_parser("check", help="检查公共字段，不校验实验私有协议")
    check.add_argument("path", help="YAML 配置文件")
    check.add_argument(
        "--common-only",
        action="store_true",
        required=True,
        help="明确只检查公共字段；私有参数必须由实验入口校验",
    )
    run = groups.add_parser("run", help="本地运行记录")
    run_commands = run.add_subparsers(dest="command", required=True)
    inspect = run_commands.add_parser("inspect", help="查看已有 run，不修改执行状态")
    inspect.add_argument("directory", help="run 目录")
    inspect.add_argument("--json", action="store_true", help="输出完整的结构化记录")
    tracking = groups.add_parser("tracking", help="显式同步已结束的 run")
    tracking_commands = tracking.add_subparsers(dest="command", required=True)
    sync = tracking_commands.add_parser("sync", help="将允许的本地结果同步到 MLflow")
    sync.add_argument("directory", help="已结束且显式启用 MLflow 的 run 目录")
    sync.add_argument(
        "--params-metrics",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="启用/关闭参数与指标同步；省略时沿用该 run 的上次策略或初始配置",
    )
    sync.add_argument(
        "--artifacts",
        choices=("off", "direct", "proxy"),
        default=None,
        help="关闭文件归档、直连 OSS/S3 或经 MLflow 代理；策略会记录以供重试",
    )
    return parser


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.group == "config":
            config = load_config(args.path)
            _print_json(
                {
                    "validation_scope": config.validation_scope,
                    "message": "公共配置有效；实验参数尚未校验，不能据此启动实验。",
                    "config": config.snapshot(),
                }
            )
        elif args.group == "run":
            record = read_run(args.directory)
            if args.json:
                _print_json(record)
            else:
                print(f"Run: {record['metadata']['run_id']}")
                print(f"实验: {record['metadata']['experiment']['id']}")
                print(f"执行状态: {record['status']['status']}")
                tracking = record["tracking"]
                print(f"同步状态: {tracking['status']}")
                if "policy" in tracking:
                    print(f"参数/指标: {tracking.get('params_metrics_status', 'pending')}")
                    print(
                        f"文件归档: {tracking.get('artifacts_status', 'pending')} "
                        f"({tracking['policy']['artifacts']})"
                    )
                print(f"指标记录: {len(record['metrics'])}")
                print(f"登记产物: {len(record['manifest']['artifacts'])}")
                print(f"目录: {record['run_dir']}")
                if record["status"]["status"] == "running":
                    print("该 run 仍运行中或未完整收尾；此命令不会把它标记为成功。")
        elif args.group == "tracking":
            from reladdr.runtime.tracking import TrackingError, sync_run

            try:
                # SDK 在 HTTP 服务收尾时会向 stdout 打印链接；CLI stdout 始终只输出 JSON。
                with redirect_stdout(sys.stderr):
                    state = sync_run(
                        args.directory, params_metrics=args.params_metrics, artifacts=args.artifacts
                    )
            except TrackingError as exc:
                print(f"同步失败：{exc}", file=sys.stderr)
                return 1
            _print_json(state)
        return 0
    except ConfigError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
    except (OSError, ValueError, KeyError, TypeError):
        # 不回显原始坏文件、异常输入或潜在认证值。
        print("无法读取完整的运行记录，请检查目录、文件和记录版本。", file=sys.stderr)
    return 1
