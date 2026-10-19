"""提供与模型及科学协议解耦的公共配置、本地执行和结果读取接口。"""

from reladdr.runtime.config import ConfigError, LoadedConfig, load_config, resolve_path
from reladdr.runtime.context import RunContext, read_run

__all__ = ["ConfigError", "LoadedConfig", "RunContext", "load_config", "read_run", "resolve_path"]
