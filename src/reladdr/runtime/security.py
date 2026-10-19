"""保护配置与运行记录：拒绝内嵌凭据，并遮盖已批准认证变量的实际值。"""

import os
from collections.abc import Mapping

_SECRET_FIELDS = {
    "password",
    "passwd",
    "secret",
    "client_secret",
    "api_key",
    "api_token",
    "access_token",
    "refresh_token",
    "private_key",
    "credentials",
    "tracking_uri",
    "mlflow_tracking_uri",
    "mlflow_tracking_password",
    "mlflow_tracking_username",
    "mlflow_tracking_token",
}
# 只读取明确的认证变量，不遍历、保存整个进程环境。
_AUTH_ENV = (
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
    "MLFLOW_TRACKING_TOKEN",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
)


def reject_secret_fields(value: object, ancestors: frozenset[int] = frozenset()) -> None:
    """检查结构中的敏感字段；字段值不出现在错误消息里。"""
    if not isinstance(value, (Mapping, list, tuple)):
        return
    if id(value) in ancestors:
        raise ValueError("配置与记录不支持循环引用")
    ancestors = ancestors | {id(value)}
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("配置与记录的字段名必须是字符串")
            if key.lower().replace("-", "_") in _SECRET_FIELDS:
                raise ValueError("请通过批准的环境变量提供凭据或服务地址，不写入配置与记录")
            reject_secret_fields(child, ancestors)
    else:
        for child in value:
            reject_secret_fields(child, ancestors)


def redact_text(text: str) -> str:
    """遮盖已知认证值；调用者仍须避免把其他私有数据写入日志。"""
    secrets = {os.environ[name] for name in _AUTH_ENV if os.environ.get(name)}
    for secret in sorted(secrets, key=len, reverse=True):
        text = text.replace(secret, "[REDACTED]")
    return text
