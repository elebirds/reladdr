"""提供独立于模型与科学协议的公共配置和运行记录能力。"""

from reladdr.runtime.config import ConfigError, LoadedConfig, load_config, resolve_path

__all__ = ["ConfigError", "LoadedConfig", "load_config", "resolve_path"]
