"""封装本地 Qwen2-VL Instruct 的 BF16 前向、预处理与颜色读出，不在导入时加载 GPU 栈。"""

import time
from pathlib import Path


def candidate_token_ids(
    tokenizer, text: str, candidate_aliases: dict[str, list[str]]
) -> dict[str, list[int]]:
    prefix = tokenizer.encode(text, add_special_tokens=False)
    result = {}
    for label, words in candidate_aliases.items():
        aliases = []
        for word in words:
            combined = tokenizer.encode(text + word, add_special_tokens=False)
            # 实际提示边界上的单 token 契约必须核对，不能静默只取多 token 答案的首项。
            if combined[: len(prefix)] != prefix or len(combined) != len(prefix) + 1:
                raise ValueError(f"候选 {word} 在当前读出边界不是单个 token")
            aliases.append(combined[-1])
        result[label] = sorted(set(aliases))
    flattened = [token for ids in result.values() for token in ids]
    if len(flattened) != len(set(flattened)):
        raise ValueError("不同候选的 token 不能重合")
    return result


class Qwen2VLAdapter:
    def __init__(self, model_path: Path, *, max_new_tokens: int = 8):
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError("GPU 调试需要可用 CUDA，不自动退回 CPU 或量化")
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            use_fast=False,
            min_pixels=448 * 448,
            max_pixels=448 * 448,
        )
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": "cuda:0"},
        ).eval()
        self.model.requires_grad_(False)
        self.metadata = {
            "model_path": str(model_path),
            "dtype": "bfloat16",
            "quantized": False,
            "attention_backend": "sdpa",
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "processor_fast": False,
            "min_pixels": 448 * 448,
            "max_pixels": 448 * 448,
            "total_device_bytes": torch.cuda.get_device_properties(0).total_memory,
        }

    def predict(self, image, prompt: str, *, candidate_aliases: dict[str, list[str]]) -> dict:
        torch = self.torch
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ids = candidate_token_ids(self.processor.tokenizer, text, candidate_aliases)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to("cuda:0")
        image_tokens = int((inputs.input_ids == self.model.config.image_token_id).sum().item())
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_logits=True,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        generated = output.sequences[0, inputs.input_ids.shape[-1] :]
        answer = self.processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
        logits = output.logits[0][0].float()
        log_z = torch.logsumexp(logits, dim=0)
        # 候选组由实验定义；合并组内别名，并保留相对完整词表的概率，不做候选内归一化。
        log_probs = {
            color: float((torch.logsumexp(logits[token_ids], dim=0) - log_z).item())
            for color, token_ids in ids.items()
        }
        return {
            "answer_text": answer,
            "candidate_log_probs": log_probs,
            "candidate_token_ids": ids,
            "input_tokens": int(inputs.input_ids.shape[-1]),
            "visual_tokens": image_tokens,
            "image_grid_thw": inputs.image_grid_thw.tolist(),
            "generation_tokens": int(generated.numel()),
            "inference_wall_time_s": elapsed,
        }

    def peak_memory(self) -> dict:
        return {
            "max_allocated_bytes": self.torch.cuda.max_memory_allocated(0),
            "max_reserved_bytes": self.torch.cuda.max_memory_reserved(0),
        }
