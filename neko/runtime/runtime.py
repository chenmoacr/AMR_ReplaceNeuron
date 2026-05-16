from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import (
    AutoTokenizer,
    Gemma4ForConditionalGeneration,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)


class AntiStutter3Plus(LogitsProcessor):
    """禁止任意 token 形成 3+ 连续 (last 2 tokens 相同则禁止下一个相同 token).
    Generic, 不依赖具体 token id - 同时防真喵 stutter 和 fake-meow (蛋糕/星星) cascade."""
    def __call__(self, input_ids, scores):
        if input_ids.shape[1] < 2:
            return scores
        last = input_ids[:, -1]
        prev = input_ids[:, -2]
        same = (last == prev)  # [batch]
        # 把 last_token 的 logit 设 -inf 仅对 same==True 的 batch
        batch_idx = torch.where(same)[0]
        if batch_idx.numel() > 0:
            scores[batch_idx, last[batch_idx]] = float("-inf")
        return scores

from steering import ClampGate, install_clamp_hooks, install_segment_gate, remove_hooks

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class _StopOnEvent(StoppingCriteria):
    def __init__(self, event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return self.event is not None and self.event.is_set()


class ClampChatRuntime:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.model_path = self.config.get("model_path", os.environ.get("GEMMA_PATH", "/path/to/gemma-4-E2B-it"))
        self.device = self.config.get("device", "cuda:0")
        self.max_new_tokens = int(self.config.get("max_new_tokens", 900))
        self.known_neurons = self.config.get("known_neurons", [])
        self.presets = self.config.get("presets", [])
        self.region_lookup = {n["id"]: n.get("region", "answer") for n in self.known_neurons}

        self.tokenizer = None
        self.model = None
        self.layers = None
        self._eoc_id = None

        # 猫娘神经元 (整列权重重写, 与 shift 注入是独立机制)
        self.meow_signature_path = Path(__file__).resolve().parent / "meow_signature.pt"
        self._meow_sig = None
        self._meow_backup = None  # {"gate": tensor, "up": tensor, "down": tensor}

    def load(self):
        if self.model is not None:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = Gemma4ForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        for attr in ("vision_tower", "audio_tower", "multi_modal_projector",
                     "embed_vision", "embed_audio"):
            if hasattr(self.model.model, attr):
                setattr(self.model.model, attr, None)
        self.layers = self.model.model.language_model.layers
        self._eoc_id = self.tokenizer.convert_tokens_to_ids("<channel|>")

    def _load_meow_signature(self):
        if self._meow_sig is not None:
            return self._meow_sig
        if not self.meow_signature_path.exists():
            raise FileNotFoundError(
                f"猫娘 signature 未生成: {self.meow_signature_path}\n"
                f"先跑: python chat/precompute_meow_signature.py"
            )
        self._meow_sig = torch.load(self.meow_signature_path, map_location="cpu",
                                    weights_only=True)
        return self._meow_sig

    def apply_meow(self, alpha: float, beta: float = 0.0):
        """
        A (L33#6417): '句号前' 触发, W_down 推 v_meow_context (软化版).
        B (L34#12188, 当 beta>0 且 sig 含 v_just_meowed_n): '前一位是喵' 触发,
                       W_down 推 -beta * v_meow_n (砍喵 logit, 剪 stutter 回路).
        """
        if self._meow_backup is not None:
            return  # 已经应用过, 避免重复备份
        sig = self._load_meow_signature()
        L = int(sig["target_layer"])
        I = int(sig["target_index"])
        scale_g = float(sig.get("scale_g", 0.2))
        scale_u = float(sig.get("scale_u", 0.2))

        W_gate_A = self.layers[L].mlp.gate_proj.weight.data
        W_up_A = self.layers[L].mlp.up_proj.weight.data
        W_down_A = self.layers[L].mlp.down_proj.weight.data

        backup = {
            "L": L, "I": I,
            "gate_A": W_gate_A[I, :].detach().clone(),
            "up_A": W_up_A[I, :].detach().clone(),
            "down_A": W_down_A[:, I].detach().clone(),
        }

        v_disc_n = sig["v_disc_n"].to(W_gate_A.device)
        v_pos_n = sig["v_pos_n"].to(W_up_A.device)
        # 硬喵: W_down 直接推 lm_head["喵"] 方向 (回退软化版本)
        # 软化用 v_meow_context_n 会引入 cluster 邻居 fake-meow, 反而让 B 砍喵设计被几何破坏
        v_down_dir = sig["v_meow_n"].to(W_down_A.device)
        W_gate_A[I, :] = (scale_g * v_disc_n).to(W_gate_A.dtype)
        W_up_A[I, :] = (scale_u * v_pos_n).to(W_up_A.dtype)
        W_down_A[:, I] = (alpha * v_down_dir).to(W_down_A.dtype)

        # === B 神经元 ===
        v_jm = sig.get("v_just_meowed_n")
        b_layer = sig.get("b_layer")
        b_index = sig.get("b_index")
        if v_jm is not None and b_layer is not None and b_index is not None and beta > 0:
            BL = int(b_layer)
            BI = int(b_index)
            scale_g_b = float(sig.get("scale_g_b", 0.2))
            scale_u_b = float(sig.get("scale_u_b", 0.2))
            W_gate_B = self.layers[BL].mlp.gate_proj.weight.data
            W_up_B = self.layers[BL].mlp.up_proj.weight.data
            W_down_B = self.layers[BL].mlp.down_proj.weight.data

            backup["BL"] = BL
            backup["BI"] = BI
            backup["gate_B"] = W_gate_B[BI, :].detach().clone()
            backup["up_B"] = W_up_B[BI, :].detach().clone()
            backup["down_B"] = W_down_B[:, BI].detach().clone()

            v_jm_dev = v_jm.to(W_gate_B.device)
            # B 砍真喵 logit: 跟 A 完全反向. A 推 +α*v_meow_n, B 推 -β*v_meow_n
            # 两者在 vocab 上是同 token 直接 cancel, 不涉及 cluster, 无 fake-meow 邻居顶替
            v_cancel = sig["v_meow_n"].to(W_down_B.device)
            W_gate_B[BI, :] = (scale_g_b * v_jm_dev).to(W_gate_B.dtype)
            W_up_B[BI, :] = (scale_u_b * v_jm_dev).to(W_up_B.dtype)
            W_down_B[:, BI] = (-beta * v_cancel).to(W_down_B.dtype)

        self._meow_backup = backup

    def restore_meow(self):
        if self._meow_backup is None:
            return
        b = self._meow_backup
        L, I = b["L"], b["I"]
        self.layers[L].mlp.gate_proj.weight.data[I, :] = b["gate_A"]
        self.layers[L].mlp.up_proj.weight.data[I, :] = b["up_A"]
        self.layers[L].mlp.down_proj.weight.data[:, I] = b["down_A"]
        if "BL" in b:
            BL, BI = b["BL"], b["BI"]
            self.layers[BL].mlp.gate_proj.weight.data[BI, :] = b["gate_B"]
            self.layers[BL].mlp.up_proj.weight.data[BI, :] = b["up_B"]
            self.layers[BL].mlp.down_proj.weight.data[:, BI] = b["down_B"]
        self._meow_backup = None

    def _split_by_region(self, selections):
        thought, answer, always = [], [], []
        for sel in selections:
            region = self.region_lookup.get(sel["id"], "answer")
            if region == "thought":
                thought.append(sel)
            elif region == "always":
                always.append(sel)
            else:
                answer.append(sel)
        return thought, answer, always

    def generate_reply(self, history, user_text: str, steering_snapshot: dict,
                       stream_callback=None, stop_event=None):
        self.load()

        think_mode = bool(steering_snapshot.get("think_mode", False))
        messages = history + [{"role": "user", "content": user_text}]
        chat_input = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=think_mode,
        )
        input_ids = chat_input["input_ids"].to(self.device)
        attention_mask = chat_input["attention_mask"].to(self.device)

        max_new_tokens = int(steering_snapshot.get("max_new_tokens", self.max_new_tokens))
        temperature = float(steering_snapshot.get("temperature", 1.0))
        do_sample = temperature > 0
        global_alpha = float(steering_snapshot.get("global_alpha", 1.0))

        hooks = []
        meow_applied = False
        try:
            if steering_snapshot.get("meow_enabled", False):
                meow_alpha = float(steering_snapshot.get("meow_alpha", 3.0))
                meow_beta = float(steering_snapshot.get("meow_beta", 0.0))
                self.apply_meow(meow_alpha, meow_beta)
                meow_applied = True

            if steering_snapshot.get("enabled", False):
                selected = [x for x in steering_snapshot.get("neurons", []) if x.get("enabled", False)]
                if selected:
                    thought_sel, answer_sel, always_sel = self._split_by_region(selected)

                    if think_mode:
                        if thought_sel:
                            tg = ClampGate(allow=True)
                            hooks += install_clamp_hooks(self.layers, thought_sel, global_alpha, gate=tg)
                            hooks.append(install_segment_gate(self.model, tg, self._eoc_id, on_seen_set_to=False))
                        if answer_sel:
                            ag = ClampGate(allow=False)
                            hooks += install_clamp_hooks(self.layers, answer_sel, global_alpha, gate=ag)
                            hooks.append(install_segment_gate(self.model, ag, self._eoc_id, on_seen_set_to=True))
                        if always_sel:
                            hooks += install_clamp_hooks(self.layers, always_sel, global_alpha, gate=None)
                    else:
                        merged = answer_sel + always_sel
                        if merged:
                            hooks += install_clamp_hooks(self.layers, merged, global_alpha, gate=None)

            gen_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                top_p=1.0,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
            # 防 stutter (符号层): 禁止任何 token 形成 3+ 连续
            # 同时治真喵 cascade 和 fake-meow (蛋糕/星星) cascade, 不需要枚举 cluster
            if meow_applied and steering_snapshot.get("meow_block_repeat", True):
                gen_kwargs["logits_processor"] = LogitsProcessorList([AntiStutter3Plus()])
            if stop_event is not None:
                gen_kwargs["stopping_criteria"] = StoppingCriteriaList([_StopOnEvent(stop_event)])

            if stream_callback is not None:
                streamer = TextIteratorStreamer(
                    self.tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=False,
                )
                gen_kwargs["streamer"] = streamer

                gen_error = {"e": None}

                def _gen_in_thread():
                    try:
                        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                            self.model.generate(**gen_kwargs)
                    except Exception as e:
                        gen_error["e"] = e

                gen_thread = threading.Thread(target=_gen_in_thread, daemon=True)
                gen_thread.start()
                collected = []
                try:
                    for new_text in streamer:
                        collected.append(new_text)
                        try:
                            stream_callback(new_text)
                        except Exception:
                            pass
                finally:
                    gen_thread.join()
                if gen_error["e"] is not None:
                    raise gen_error["e"]
                return "".join(collected)

            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                out = self.model.generate(**gen_kwargs)
            gen_ids = out[0][input_ids.shape[1]:]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
            return text
        finally:
            if hooks:
                remove_hooks(hooks)
            if meow_applied:
                self.restore_meow()

    def get_preset(self, preset_name: str):
        for p in self.presets:
            if p.get("name") == preset_name:
                return p
        return None
