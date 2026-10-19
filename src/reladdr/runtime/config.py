"""读取公共 YAML 配置，将实验参数的完整校验留给对应实验。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, field_validator

from reladdr.runtime.security import redact_text, reject_secret_fields


class ConfigError(ValueError):
    """可直接向用户展示、不会携带原始凭据值的配置错误。"""


class StrictModel(BaseModel):
    """公共配置拒绝未知字段和隐式类型转换。"""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, hide_input_in_errors=True)


class ExperimentIdentity(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    protocol_version: str = Field(min_length=1, pattern=r"\S")
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class RunSettings(StrictModel):
    seed: int = Field(ge=0, le=2**32 - 1)
    output_root: Path = Path("runs")
    label: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @field_validator("output_root", mode="before")
    @classmethod
    def path_from_yaml(cls, value: object) -> object:
        if isinstance(value, str) and value.strip():
            return Path(value)
        if isinstance(value, Path):
            return value
        raise ValueError("output_root 必须是非空路径")


class TrackingSettings(StrictModel):
    mode: Literal["local", "mlflow"] = "local"
    experiment_name: str | None = Field(default=None, min_length=1)

    @field_validator("experiment_name")
    @classmethod
    def nonblank_name(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("experiment_name 不能为空白")
        return value


class ConfigDocument(StrictModel):
    schema_version: Literal[1] = 1
    experiment: ExperimentIdentity
    run: RunSettings
    tracking: TrackingSettings = Field(default_factory=TrackingSettings)

    @field_validator("schema_version", mode="before")
    @classmethod
    def exact_schema_version(cls, value: object) -> int:
        # Literal[1] 本身会接受 True/1.0；版本号必须明确为整数，不能靠相等关系转换。
        if type(value) is not int or value != 1:
            raise ValueError("schema_version 必须是整数 1")
        return value


class _UniqueSafeLoader(yaml.SafeLoader):
    """安全加载 YAML，并拒绝会被普通加载器静默覆盖的重复字段。"""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        self.flatten_mapping(node)
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ConfigError("YAML 字段名必须是字符串")
            if key in result:
                raise ConfigError("YAML 存在重复字段")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def resolve_path(value: Path | str, base_dir: Path) -> Path:
    """配置路径相对配置文件解析；实验自己的路径字段也可复用此函数。"""
    path = Path(value).expanduser()
    return (base_dir / path).resolve() if not path.is_absolute() else path.resolve()


def _validation_message(exc: ValidationError) -> str:
    # 不输出 Pydantic 默认附带的输入值，也不把自定义校验上下文序列化。
    return redact_text(
        "; ".join(
            f"{'.'.join(map(str, error['loc']))}: {error['msg']}"
            for error in exc.errors(include_input=False, include_context=False, include_url=False)
        )
    )


@dataclass(frozen=True)
class LoadedConfig:
    """持有公共配置与可选的实验参数模型，并明确实际完成的校验范围。"""

    source: Path
    document: ConfigDocument
    parameters: BaseModel | None = None

    @property
    def validation_scope(self) -> str:
        return "experiment" if self.parameters is not None else "common_only"

    def snapshot(self) -> dict:
        result = self.document.model_dump(mode="json")
        if self.parameters is not None:
            result["experiment"]["parameters"] = self.parameters.model_dump(mode="json")
        reject_secret_fields(result)
        return result


def load_config(
    path: Path | str, *, parameter_model: type[BaseModel] | None = None
) -> LoadedConfig:
    """加载公共配置；真实实验入口必须传入自己的参数模型。"""
    source = Path(path).expanduser().resolve()
    try:
        raw = yaml.load(source.read_text(encoding="utf-8"), Loader=_UniqueSafeLoader)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        location = f"，第 {mark.line + 1} 行、第 {mark.column + 1} 列" if mark else ""
        raise ConfigError(f"无效 YAML{location}") from None
    except UnicodeError:
        raise ConfigError("配置文件必须使用 UTF-8 编码") from None
    except OSError:
        raise ConfigError("无法读取配置文件") from None
    try:
        reject_secret_fields(raw)
        document = ConfigDocument.model_validate(raw)
        if document.tracking.mode == "mlflow" and not document.tracking.experiment_name:
            raise ConfigError("启用 MLflow 时必须提供 tracking.experiment_name")
        document = document.model_copy(
            update={
                "run": document.run.model_copy(
                    update={"output_root": resolve_path(document.run.output_root, source.parent)}
                )
            }
        )
        parameters = None
        if parameter_model is not None:
            if parameter_model.model_config.get("extra") != "forbid":
                raise ConfigError("实验参数模型必须声明 extra='forbid'")
            parameters = parameter_model.model_validate(document.experiment.parameters, strict=True)
        loaded = LoadedConfig(source, document, parameters)
        loaded.snapshot()
        return loaded
    except ValidationError as exc:
        raise ConfigError(_validation_message(exc)) from None
    except ValueError as exc:
        if isinstance(exc, ConfigError):
            raise
        raise ConfigError(str(exc)) from None
