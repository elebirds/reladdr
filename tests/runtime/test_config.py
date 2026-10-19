"""验证公共配置、实验私有校验、路径语义及敏感字段边界。"""

from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from reladdr.runtime import ConfigError, load_config
from reladdr.runtime.config import StrictModel


@pytest.fixture
def config_file(tmp_path):
    def write(extra="", *, parameters="{}"):
        path = tmp_path / "config.yaml"
        path.write_text(
            "experiment:\n  id: E01\n  protocol_version: '1'\n"
            f"  parameters: {parameters}\nrun:\n  seed: 7\n  output_root: results\n" + extra,
            encoding="utf-8",
        )
        return path

    return write


def test_common_scope_resolves_relative_to_config(config_file, monkeypatch, tmp_path):
    source = config_file(parameters="{unimplemented_parameter: 3}")
    monkeypatch.chdir(tmp_path.parent)
    loaded = load_config(source)
    assert loaded.document.run.output_root == source.parent / "results"
    assert loaded.validation_scope == "common_only"
    assert loaded.snapshot()["experiment"]["parameters"] == {"unimplemented_parameter": 3}
    assert loaded.snapshot()["tracking"]["mode"] == "local"


def test_experiment_owns_parameter_schema(config_file):
    class PrivateSettings(StrictModel):
        samples: int = Field(gt=0)
        split: str = "discovery"

    loaded = load_config(config_file(parameters="{samples: 5}"), parameter_model=PrivateSettings)
    assert loaded.parameters.samples == 5
    assert loaded.validation_scope == "experiment"
    assert loaded.snapshot()["experiment"]["parameters"] == {"samples": 5, "split": "discovery"}
    with pytest.raises(ConfigError, match="Extra inputs"):
        load_config(
            config_file(parameters="{samples: 5, donor: other}"), parameter_model=PrivateSettings
        )


@pytest.mark.parametrize("extra", ["run_typo: 1\n", "tracking:\n  mode: invalid\n"])
def test_unknown_fields_and_enums_fail(config_file, extra):
    with pytest.raises(ConfigError):
        load_config(config_file(extra))


@pytest.mark.parametrize("seed", ["true", "'7'", "-1", "4294967296"])
def test_seed_is_not_coerced(config_file, seed):
    source = config_file()
    source.write_text(source.read_text().replace("seed: 7", f"seed: {seed}"))
    with pytest.raises(ConfigError):
        load_config(source)


@pytest.mark.parametrize(
    "parameters",
    [
        "{password: sensitive-value}",
        "{nested: [{api_key: sensitive-value}]}",
        "{tracking_uri: 'https://user:password@service/'}",
    ],
)
def test_secrets_are_rejected_without_echo(config_file, parameters):
    with pytest.raises(ConfigError) as error:
        load_config(config_file(parameters=parameters))
    assert "sensitive-value" not in str(error.value)
    assert "https://" not in str(error.value)


def test_visual_token_fields_are_not_credentials(config_file):
    assert load_config(config_file(parameters="{token_budget: 256}"))


def test_duplicate_yaml_fields_fail(config_file):
    with pytest.raises(ConfigError, match="重复字段"):
        load_config(config_file("tracking: {mode: local, mode: mlflow}\n"))


@pytest.mark.parametrize(
    "content",
    [
        "password: [do-not-echo-secret",
        "!!python/object/apply:os.system ['echo unsafe']",
        "[]",
        "null",
        "experiment: &recursive {parameters: *recursive}",
    ],
)
def test_invalid_and_unsafe_yaml_fail(tmp_path, content):
    source = tmp_path / "invalid.yaml"
    source.write_text(content)
    with pytest.raises(ConfigError) as error:
        load_config(source)
    assert "do-not-echo-secret" not in str(error.value)


def test_missing_file_has_clean_error(tmp_path):
    with pytest.raises(ConfigError, match="无法读取"):
        load_config(tmp_path / "missing")


def test_tracking_requires_experiment_name(config_file):
    with pytest.raises(ConfigError, match="experiment_name"):
        load_config(config_file("tracking: {mode: mlflow}\n"))
    loaded = load_config(config_file("tracking: {mode: mlflow, experiment_name: reladdr-E01}\n"))
    assert loaded.document.tracking.experiment_name == "reladdr-E01"


def test_private_schema_cannot_silently_ignore_typos(config_file):
    class UnsafeSettings(BaseModel):
        pass

    with pytest.raises(ConfigError, match="extra='forbid'"):
        load_config(config_file(), parameter_model=UnsafeSettings)


def test_private_path_schema_is_not_guessed_by_runtime(config_file):
    class PrivateSettings(StrictModel):
        input_file: Path

    source = config_file(parameters="{input_file: relative.json}")
    # 路径转换由实验自己的 schema 决定，公共层不会根据字段名偷偷改值。
    with pytest.raises(ConfigError):
        load_config(source, parameter_model=PrivateSettings)
