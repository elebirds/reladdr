"""验证结果曲线使用声明切片并生成可读图，不接受缺失数据的空图。"""

import pytest
from PIL import Image

from reladdr.evaluation.curves import plot_patching_curves


def test_curves_render_selected_real_shape_and_reject_empty(tmp_path):
    rows = [
        {
            "split": "discovery",
            "setting": "squares",
            "baseline_correct_subset": False,
            "position": "readout",
            "layer": layer,
            "n": 16,
            "outcomes": {
                name: {"probability": {"estimate": 0.2 + layer / 100, "ci95": [0.1, 0.4]}}
                for name in ("clean", "address", "attribute")
            },
        }
        for layer in range(3)
    ]
    path = tmp_path / "plot.png"
    plot_patching_curves(
        {"curves": rows}, path, title="Controlled test only", split="discovery", setting="squares"
    )
    assert Image.open(path).size == (600, 480)
    with pytest.raises(ValueError):
        plot_patching_curves(
            {"curves": rows}, tmp_path / "empty.png", title="Empty", split="test", setting="squares"
        )
