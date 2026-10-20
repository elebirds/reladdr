"""为 Qwen2-VL 提供真实 token 映射、无缓存原生前向及可恢复的残差/组件干预。"""

import time
from collections import defaultdict
from dataclasses import dataclass

from reladdr.interventions.hooks import replace_rows, select_rows, temporary_hooks
from reladdr.models.qwen2_vl import Qwen2VLAdapter


def find_phrase(ids, tokenizer, phrase):
    """在实际展开后的输入中匹配完整词组，避免把像素 patch 序号当语言 token 序号。"""
    matches = set()
    for text in (phrase, " " + phrase):
        part = tokenizer.encode(text, add_special_tokens=False)
        if not part:
            continue
        for start in range(len(ids) - len(part) + 1):
            if ids[start : start + len(part)] == part:
                matches.add(tuple(range(start, start + len(part))))
    if not matches:
        raise ValueError(f"实际序列中没有匹配词组：{phrase}")
    # 系统提示可能含同词；问题中的最后一次完整匹配是本协议约定的语义位置。
    return max(matches, key=lambda values: (values[-1], len(values)))


@dataclass
class PreparedInput:
    inputs: object
    text: str
    candidate_ids: dict[str, list[int]]
    roles: dict[str, tuple[int, ...]]
    token_record: dict


class Qwen2NativeAdapter(Qwen2VLAdapter):
    def __init__(self, model_path, *, system_prompt, image_size=336):
        super().__init__(
            model_path, max_new_tokens=1, system_prompt=system_prompt, image_size=image_size
        )
        self.layers = self.model.model.language_model.layers
        config = self.model.config.text_config
        self.n_layers = len(self.layers)
        self.n_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.n_heads
        self.forward_calls = 0
        self.metadata.update(
            {
                "n_layers": self.n_layers,
                "n_query_heads": self.n_heads,
                "n_kv_heads": config.num_key_value_heads,
                "forward_protocol": "full_prefill_no_cache; native_final_norm_and_lm_head",
            }
        )

    def prepare(self, image, prompt, *, candidate_aliases, reference=None, relation=None):
        inputs, text, candidates = self.encode(image, prompt, candidate_aliases=candidate_aliases)
        tokenizer = self.processor.tokenizer
        ids = inputs.input_ids[0].tolist()
        user_end = [
            i
            for i, token in enumerate(ids)
            if token == tokenizer.convert_tokens_to_ids("<|im_end|>")
        ][-1]
        visual = tuple(
            i for i, token in enumerate(ids) if token == self.model.config.image_token_id
        )
        roles = {
            "question_end": find_phrase(ids, tokenizer, prompt.rstrip().split()[-1])[-1:],
            "readout": (len(ids) - 1,),
            "visual": visual,
        }
        roles["text_suffix"] = tuple(range(visual[-1] + 1, len(ids)))
        if reference:
            roles["reference"] = find_phrase(ids, tokenizer, reference)
        if relation:
            roles["relation"] = find_phrase(ids, tokenizer, relation)
        record = {
            "user_message_end_index": user_end,
            "input_tokens": len(ids),
            "visual_tokens": len(visual),
            "image_grid_thw": inputs.image_grid_thw.tolist(),
            "roles": {key: list(value) for key, value in roles.items()},
            "text_tokens": [
                {"index": i, "id": ids[i], "text": tokenizer.decode([ids[i]])}
                for i in range(visual[-1] + 1, len(ids))
            ],
            "candidate_token_ids": candidates,
        }
        return PreparedInput(inputs, text, candidates, roles, record)

    def _module(self, layer, component):
        block = self.layers[layer]
        if component == "residual":
            return block, "post"
        if component == "attention":
            return block.self_attn, "post"
        if component == "mlp":
            return block.mlp, "post"
        if component == "heads":
            return block.self_attn.o_proj, "pre"
        return getattr(
            block.self_attn, {"query": "q_proj", "key": "k_proj", "value": "v_proj"}[component]
        ), "post"

    def forward(self, prepared, *, capture=(), patches=None):
        torch = self.torch
        patches = patches or {}
        grouped = defaultdict(list)
        for site in dict.fromkeys([*capture, *patches]):
            if site.layer >= self.n_layers:
                raise ValueError("层编号超出模型范围")
            grouped[site.layer, site.component].append(site)
        saved = {}
        registrations = []
        for (layer, component), sites in grouped.items():
            module, kind = self._module(layer, component)

            def edit(tensor, selected_sites=sites):
                for site in selected_sites:
                    if site in patches:
                        tensor = replace_rows(tensor, site, patches[site], self.head_dim)
                    if site in capture:
                        saved[site] = (
                            select_rows(tensor, site, self.head_dim).detach().cpu().clone()
                        )
                return tensor

            if kind == "pre":

                def hook(_module, args, edit=edit):
                    return (edit(args[0]), *args[1:])
            else:

                def hook(_module, _args, output, edit=edit):
                    if isinstance(output, tuple):
                        return (edit(output[0]), *output[1:])
                    return edit(output)

            registrations.append((module, kind, hook))
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode(), temporary_hooks(registrations):
            # 先计算模型真实的最终归一化状态，再只对最后位置执行原生 lm_head。
            # 不生成整段回答、不物化全部 token 的巨大词表矩阵；不复用受干预的旧缓存。
            output = self.model.model(**prepared.inputs, use_cache=False, return_dict=True)
            logits = self.model.lm_head(output.last_hidden_state[:, -1:, :])[0, 0].float()
            log_z = torch.logsumexp(logits, dim=0)
            log_probs = {
                color: float((torch.logsumexp(logits[token_ids], dim=0) - log_z).item())
                for color, token_ids in prepared.candidate_ids.items()
            }
            top = torch.topk(logits, 5)
            raw_id = int(top.indices[0].item())
            top_tokens = [
                {
                    "id": int(token),
                    "text": self.processor.tokenizer.decode([int(token)]),
                    "logit": float(value),
                }
                for token, value in zip(top.indices.tolist(), top.values.tolist(), strict=True)
            ]
        self.forward_calls += 1
        torch.cuda.synchronize()
        prediction = {
            "answer_text": self.processor.tokenizer.decode([raw_id]).strip(),
            "candidate_prediction": max(log_probs, key=log_probs.get),
            "candidate_log_probs": log_probs,
            "top_tokens": top_tokens,
            "inference_wall_time_s": time.perf_counter() - started,
        }
        return prediction, saved, logits.detach().cpu()
