# =============================================================================
# FILE: stage1_qwen.py
# CATEGORY: ARCH
# PURPOSE: Stage 1 visual grounding built on Qwen2.5-VL with optional LoRA adapters.
# DEPENDENCIES: None
# USED BY: benchmark_runtime.py, check_trainable.py, pipeline.py, run_dev_inference.py, run_single_inference.py, train_joint.py, validate_pipeline_contracts.py, visualize_val_predictions.py
# KEY CLASSES/FUNCTIONS: QwenGrounder, parse_detections, _is_valid_detection
# LAST MODIFIED: 2026-03-21T13:01:59.772992+00:00
# SNAPSHOT NOTES: no major inline issues detected during snapshot generation
# =============================================================================
"""
trifoodnet/models/stage1_qwen.py
─────────────────────────────────
Stage 1 — Qwen2.5-VL-3B visual grounding with LoRA fine-tuning.

The model receives a dish image + text prompt and outputs a JSON list of
{ "box": [x1,y1,x2,y2], "label": "<coarse_name>" } per food item.

LoRA targets q_proj + v_proj in all attention layers (rank 16).
The rest of the weights are frozen.

Outputs
-------
parse_output(text) → List[Dict]   # parsed JSON detections
forward(...)       → CausalLMOutputWithPast  (loss computed if labels given)
"""

from __future__ import annotations
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

try:
    from peft import LoraConfig, get_peft_model, TaskType
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


# ──────────────────────────────────────────────────────────────────────────────
# Default prompt
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_PROMPT = (
    "Identify all distinct food items on this cafeteria dish plate. "
    "For each item output a JSON object with a single key: "
    '"box": [x1, y1, x2, y2] (pixel coordinates, top-left to bottom-right). '
    "Return ONLY a JSON array. No extra text."
)


# ──────────────────────────────────────────────────────────────────────────────
# Model wrapper
# ──────────────────────────────────────────────────────────────────────────────

# --- Snapshot note: Core Stage 1 module: language-supervised visual grounding and JSON box generation. ---
class QwenGrounder(nn.Module):
    """
    Qwen2.5-VL-3B wrapped with optional LoRA for fine-tuning.

    Parameters
    ----------
    model_name           : HuggingFace model id
    lora_r               : LoRA rank (0 = disabled)
    lora_alpha           : LoRA scaling alpha
    lora_dropout         : dropout on LoRA weights
    lora_target_modules  : which projection layers to adapt
    gradient_checkpointing: enables activation checkpointing
    """

    def __init__(
        self,
        model_name:           str,
        lora_r:               int = 16,
        lora_alpha:           int = 32,
        lora_dropout:         float = 0.05,
        lora_target_modules:  Sequence[str] = ("q_proj", "v_proj"),
        use_rslora:           bool = False,
        torch_dtype:          Optional[torch.dtype] = None,
        device_map:           Optional[str | Dict] = None,
        device:              Optional[str | torch.device] = None,
        gradient_checkpointing: bool = False,
        quantization_config:  Optional[Any] = None,
        debug:                bool = False,
    ):
        super().__init__()
        self.debug = bool(debug)

        model_dtype = torch_dtype or (torch.float16 if torch.cuda.is_available() else torch.float32)
        resolved_device = torch.device(device) if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        resolved_device_map = device_map if device_map is not None else (str(resolved_device) if resolved_device.type == "cuda" else None)

        load_kwargs = {"torch_dtype": model_dtype}
        if resolved_device_map is not None:
            load_kwargs["device_map"] = resolved_device_map
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
            # Force GPU-only to avoid bitsandbytes CPU-offload errors on RTX 5090
            load_kwargs["device_map"] = "cuda:0"

        self.base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            **load_kwargs,
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

        if gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable()
            if hasattr(self.base_model.config, "use_cache"):
                self.base_model.config.use_cache = False

        # Apply LoRA
        if lora_r > 0:
            if not HAS_PEFT:
                raise ImportError("Install `peft` to use LoRA: pip install peft")
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=list(lora_target_modules),
                use_rslora=use_rslora,
                bias="none",
            )
            self.model = get_peft_model(self.base_model, lora_cfg)
        else:
            self.model = self.base_model

        if quantization_config is None and resolved_device.type != "cuda":
            self.model.to(resolved_device)

    # ── parameter reporting ───────────────────────────────────────────────────

    def print_trainable_parameters(self):
        if HAS_PEFT:
            self.model.print_trainable_parameters()
        else:
            total  = sum(p.numel() for p in self.model.parameters())
            train  = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Trainable: {train:,} / {total:,} ({100*train/total:.2f}%)")

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Standard HF forward — returns CausalLMOutputWithPast.
        Loss is computed automatically when `labels` is provided.
        """
        # Move all inputs to same device as model
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(device)
        if labels is not None:
            labels = labels.to(device)

        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            **kwargs,
        )

    # ── inference helpers ─────────────────────────────────────────────────────

    @torch.inference_mode()
    def generate_detections(
        self,
        images,                         # PIL images or list of PIL images
        prompt: str = DEFAULT_PROMPT,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> List[List[Dict]]:
        """
        Run inference on a batch of PIL images.
        Returns list (one per image) of parsed detection dicts.
        """
        if not isinstance(images, list):
            images = [images]

        messages_batch = [
            [{"role": "user", "content": [
                {"type": "image",  "image": img},
                {"type": "text",   "text":  prompt},
            ]}]
            for img in images
        ]

        inputs_batch = [
            self.processor.apply_chat_template(msg, add_generation_prompt=True)
            for msg in messages_batch
        ]

        inputs = self.processor(
            text=inputs_batch,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        gen_kwargs: Dict[str, Any] = {"max_new_tokens": max_new_tokens}
        if temperature == 0.0:
            gen_kwargs.update({"do_sample": False})
        else:
            gen_kwargs.update({"do_sample": True, "temperature": temperature})

        output_ids = self.model.generate(**inputs, **gen_kwargs)
        # Slice off the input tokens to get only generated text
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        texts = self.processor.batch_decode(generated, skip_special_tokens=True)
        if self.debug and texts:
            print(f"--- Qwen Raw Output ---\n{texts[0]}\n-----------------------")

        return [parse_detections(t) for t in texts]

    # ── checkpointing ─────────────────────────────────────────────────────────

    def save_lora(self, path: str):
        """Save only the LoRA adapter weights (small, ~few MB)."""
        if HAS_PEFT:
            self.model.save_pretrained(path)
        else:
            torch.save(self.model.state_dict(), path)

    def load_lora(self, path: str):
        if HAS_PEFT:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.base_model, path)
        else:
            self.model.load_state_dict(torch.load(path, map_location="cpu"))


# ──────────────────────────────────────────────────────────────────────────────
# JSON parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_detections(text: str) -> List[Dict]:
    """
    Extract JSON array from model output.
    Returns list of {"box": [x1,y1,x2,y2], "label": str} dicts.
    Falls back to [] on any parse failure.
    """
    # Strip markdown fences if present
    text = re.sub(r"```json|```", "", text).strip()

    # Try direct parse
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [d for d in data if _is_valid_detection(d)]
        return []
    except json.JSONDecodeError:
        pass

    # Fallback: find first [...] block
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                return [d for d in data if _is_valid_detection(d)]
        except json.JSONDecodeError:
            pass

    return []


def _is_valid_detection(d: Any) -> bool:
    if not isinstance(d, dict):
        return False
    box = d.get("box")
    # Robust check: Qwen sometimes wraps the box in an extra list [[x1,y1,x2,y2]]
    if isinstance(box, (list, tuple)) and len(box) == 1 and isinstance(box[0], (list, tuple)):
        box = box[0]
        d["box"] = box # Flatten it for later stages
        
    if not (isinstance(box, (list, tuple)) and len(box) == 4):
        return False
    return True
