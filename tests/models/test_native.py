"""验证真实输入中的语义 token 匹配，不导入 CUDA 模型或下载权重。"""

import pytest

from reladdr.models.qwen2_native import find_phrase


class Tokenizer:
    def encode(self, text, *, add_special_tokens):
        return [ord(c) for c in text]


def test_phrase_matching_is_complete_and_uses_last_occurrence():
    tokenizer = Tokenizer()
    text = "blue square. The color of the square left of blue square is "
    ids = [ord(c) for c in text]
    indices = find_phrase(ids, tokenizer, "blue square")
    assert "".join(chr(ids[i]) for i in indices).strip() == "blue square"
    assert indices[-1] == text.rindex("blue square") + len("blue square") - 1
    with pytest.raises(ValueError):
        find_phrase(ids, tokenizer, "green circle")
