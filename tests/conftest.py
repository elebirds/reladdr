"""为运行层测试构造隔离配置，不注册或模拟任何科学实验入口。"""

import pytest

from reladdr.runtime.config import StrictModel, load_config


class RuntimeTestParameters(StrictModel):
    """仅声明测试上下文没有研究参数，避免绕过私有参数校验契约。"""


@pytest.fixture
def runtime_config(tmp_path):
    def create(*, tracking="local", output_root="runs", seed=19):
        source = tmp_path / "runtime.yaml"
        source.write_text(
            "# 运行层测试配置，不代表模型实验。\n"
            "experiment:\n  id: runtime_test\n  protocol_version: '1'\n  parameters: {}\n"
            f"run:\n  seed: {seed}\n  output_root: {output_root}\n"
            f"tracking:\n  mode: {tracking}\n  experiment_name: reladdr-runtime-tests\n",
            encoding="utf-8",
        )
        return load_config(source, parameter_model=RuntimeTestParameters)

    return create
