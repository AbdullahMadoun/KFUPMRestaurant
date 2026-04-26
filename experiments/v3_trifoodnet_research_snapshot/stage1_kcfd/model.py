from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch


@dataclass
class ModelConfig:
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    bf16: bool = True
    gradient_checkpointing: bool = True
    device_map: str = "auto"
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    min_pixels: int | None = None
    max_pixels: int | None = 1280 * 28 * 28
    attn_implementation: str | None = None
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ])
    unfreeze_vision: bool = True


def _resolve_device_map(value: str | None) -> Optional[str]:
    normalized = str(value or "").strip().lower()
    if normalized in {"", "none", "null", "cpu"}:
        return None
    return str(value)


def first_parameter_device(model: torch.nn.Module) -> torch.device:
    for parameter in model.parameters():
        return parameter.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _nested_attr(obj: Any, path: str) -> Optional[Any]:
    cur = obj
    for part in path.split("."):
        cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur


def find_vision_module(model: torch.nn.Module) -> Optional[torch.nn.Module]:
    candidates = (
        "visual",
        "vision_tower",
        "vision_model",
        "model.visual",
        "model.vision_tower",
        "model.vision_model",
        "base_model.model.visual",
        "base_model.model.vision_tower",
        "base_model.model.vision_model",
        "base_model.model.model.visual",
    )
    for path in candidates:
        module = _nested_attr(model, path)
        if isinstance(module, torch.nn.Module):
            return module
    return None


def unfreeze_vision_encoder(model: torch.nn.Module) -> int:
    vision = find_vision_module(model)
    if vision is None:
        return 0
    count = 0
    for parameter in vision.parameters():
        parameter.requires_grad = True
        count += parameter.numel()
    return count


def trainable_parameter_summary(model: torch.nn.Module) -> Dict[str, float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total_params": float(total),
        "trainable_params": float(trainable),
        "trainable_pct": float(100.0 * trainable / max(total, 1)),
    }


def save_vision_encoder_state(model: torch.nn.Module, path: str | Path) -> int:
    vision = find_vision_module(model)
    if vision is None:
        return 0
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu() for key, value in vision.state_dict().items()}
    torch.save({"state_dict": state}, target)
    return sum(tensor.numel() for tensor in state.values())


def load_vision_encoder_state(model: torch.nn.Module, path: str | Path, *, strict: bool = True):
    vision = find_vision_module(model)
    if vision is None:
        raise ValueError("could not find Qwen vision module on this model")
    payload = torch.load(path, map_location="cpu")
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    return vision.load_state_dict(state, strict=strict)


def load_stage1_checkpoint(
    checkpoint_dir: str | Path,
    *,
    model_config: ModelConfig | None = None,
    is_trainable: bool = False,
):
    try:
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "Stage 1 checkpoint loading requires transformers and peft. "
            "Install requirements-stage1.txt first."
        ) from exc

    checkpoint = Path(checkpoint_dir)
    manifest_path = checkpoint / "checkpoint_manifest.json"
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        import json

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = model_config or ModelConfig(model_id=str(manifest.get("base_model") or ModelConfig.model_id))
    dtype = torch.bfloat16 if config.bf16 and torch.cuda.is_available() else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )
    processor_dir = checkpoint / str(manifest.get("processor_dir", "processor"))
    adapter_dir = checkpoint / str(manifest.get("adapter_dir", "model"))
    vision_path = checkpoint / str(manifest.get("vision_encoder", "vision_encoder.pt"))
    processor_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if config.min_pixels is not None:
        processor_kwargs["min_pixels"] = config.min_pixels
    if config.max_pixels is not None:
        processor_kwargs["max_pixels"] = config.max_pixels
    processor = AutoProcessor.from_pretrained(processor_dir if processor_dir.exists() else config.model_id, **processor_kwargs)
    model_kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": _resolve_device_map(config.device_map),
        "trust_remote_code": True,
    }
    if config.attn_implementation:
        model_kwargs["attn_implementation"] = config.attn_implementation
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(config.model_id, **model_kwargs)
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=is_trainable)
    vision_params_saved = int(manifest.get("vision_params_saved", 0) or 0)
    if vision_params_saved > 0 and not vision_path.exists():
        raise FileNotFoundError(f"checkpoint manifest requires vision state, but it is missing: {vision_path}")
    if vision_path.exists():
        load_result = load_vision_encoder_state(model, vision_path, strict=vision_params_saved > 0)
        if vision_params_saved > 0 and (load_result.missing_keys or load_result.unexpected_keys):
            raise RuntimeError(
                "vision encoder checkpoint did not load cleanly: "
                f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
            )
    return model, processor


def build_model_and_processor(config: ModelConfig):
    try:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "Stage 1 training dependencies are missing. Install requirements-stage1.txt "
            "inside the training environment before building the Qwen model."
        ) from exc

    dtype = torch.bfloat16 if config.bf16 and torch.cuda.is_available() else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )
    device_map = _resolve_device_map(config.device_map)
    processor_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if config.min_pixels is not None:
        processor_kwargs["min_pixels"] = config.min_pixels
    if config.max_pixels is not None:
        processor_kwargs["max_pixels"] = config.max_pixels
    processor = AutoProcessor.from_pretrained(config.model_id, **processor_kwargs)
    model_kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": device_map,
        "trust_remote_code": True,
    }
    if config.attn_implementation:
        model_kwargs["attn_implementation"] = config.attn_implementation
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(config.model_id, **model_kwargs)
    if device_map is None:
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    vision_params = unfreeze_vision_encoder(model) if config.unfreeze_vision else 0
    return model, processor, vision_params


def tensor_batch_to_device(batch: Dict[str, Any], device: torch.device | str) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def model_input_tensors(batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    return {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}
