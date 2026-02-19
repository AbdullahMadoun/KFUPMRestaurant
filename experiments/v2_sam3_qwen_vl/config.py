"""Pipeline configuration as nested stdlib dataclasses with JSON I/O."""

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
from typing import List


@dataclass
class QwenConfig:
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    gpu_memory_utilization: float = 0.4
    max_model_len: int = 4096
    enforce_eager: bool = True
    allowed_local_media_path: str = "/root"
    temperature: float = 0.1
    max_tokens: int = 256


@dataclass
class PromptConfig:
    template: str = (
        "Identify the distinct food SERVINGS visible in this image.\n"
        "\n"
        "RULES:\n"
        "- Each visually separate portion of food = one item\n"
        "- A mixed dish (e.g., salad, stew, soup) is ONE item — do NOT list its ingredients separately\n"
        "- If two different foods sit side by side (e.g., rice next to chicken), list them as SEPARATE items\n"
        "- Do NOT include plates, bowls, cutlery, wrapping, or background\n"
        "\n"
        "Examples:\n"
        "- Rice and chicken on same plate → [\"rice\", \"chicken\"]\n"
        "- A bowl of mixed salad → [\"mixed salad\"] (NOT [\"lettuce\", \"tomato\", ...])\n"
        "- Soup with bread on the side → [\"soup\", \"bread\"]\n"
        "\n"
        "Return strictly as JSON: {\"food_items\": [\"item1\", \"item2\", ...]}\n"
        "Return ONLY the JSON."
    )
    grounding_template: str = (
        'Locate the "{label}" in this image. '
        'Return a JSON object with the bounding box: '
        '{{"bbox_2d": [x1, y1, x2, y2]}} '
        'where coordinates are pixel positions. Return ONLY the JSON.'
    )
    prompt_mode: str = "text"  # "text" or "grounded"


@dataclass
class SAM3Config:
    model_path: str = "facebook/sam3"
    confidence_threshold: float = 0.1
    fallback_thresholds: List[float] = field(default_factory=lambda: [0.05, 0.02, 0.01])
    bpe_search_paths: List[str] = field(default_factory=lambda: [
        "/root/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
    ])


@dataclass
class NMSConfig:
    max_objects: int = 5
    iou_threshold: float = 0.7


@dataclass
class VizConfig:
    draw_boxes: bool = True
    alpha: float = 0.7
    thickness: int = 3
    font_scale: float = 0.9
    font_thickness: int = 2
    color_low: List[int] = field(default_factory=lambda: [0, 50])
    color_high: List[int] = field(default_factory=lambda: [180, 255])


@dataclass
class PipelineConfig:
    device: str = "cuda"
    output_dir: str = "bold_results"
    image_extensions: List[str] = field(default_factory=lambda: [
        "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp",
    ])
    sam3_repo_path: str = "sam3"
    qwen: QwenConfig = field(default_factory=QwenConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    sam3: SAM3Config = field(default_factory=SAM3Config)
    nms: NMSConfig = field(default_factory=NMSConfig)
    viz: VizConfig = field(default_factory=VizConfig)

    def to_json(self, path: str) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: str) -> "PipelineConfig":
        data = json.loads(Path(path).read_text())
        return cls(
            device=data.get("device", "cuda"),
            output_dir=data.get("output_dir", "bold_results"),
            image_extensions=data.get("image_extensions", ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]),
            sam3_repo_path=data.get("sam3_repo_path", "sam3"),
            qwen=QwenConfig(**data["qwen"]) if "qwen" in data else QwenConfig(),
            prompt=PromptConfig(**data["prompt"]) if "prompt" in data else PromptConfig(),
            sam3=SAM3Config(**data["sam3"]) if "sam3" in data else SAM3Config(),
            nms=NMSConfig(**data["nms"]) if "nms" in data else NMSConfig(),
            viz=VizConfig(**data["viz"]) if "viz" in data else VizConfig(),
        )
