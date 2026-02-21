"""Pipeline configuration as nested stdlib dataclasses with JSON I/O."""

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
from typing import List


@dataclass
class VLMConfig:
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 4096
    enforce_eager: bool = True
    quantization: str = None  # None, "fp8", "awq" — passed to vLLM LLM()
    force_json: bool = True   # use vLLM guided decoding to force valid JSON output
    allowed_local_media_path: str = "/root"
    temperature: float = 0.2
    max_tokens: int = 512
    describe_template: str = (
        "You are analyzing a cafeteria tray. Find EVERY food and beverage item — "
        "do not miss anything. Scan the full image carefully: main dishes, sides, "
        "bread, dips, drinks, juice boxes, yogurt cups, fruit, desserts.\n"
        "\n"
        "For each item return:\n"
        '- "name": short common name (e.g. "kabsa rice", "grilled chicken", "hummus")\n'
        '- "description": visual texture ONLY — color, surface pattern, shape. '
        "Do not mention plates, positions, or containers.\n"
        '- "bbox": [x1, y1, x2, y2] absolute pixel coordinates. '
        "Box must fully cover the food with a small margin.\n"
        "\n"
        "Splitting rules:\n"
        "- Different foods = separate items (rice next to meat = 2 items)\n"
        "- Same food in two places = list twice with separate bboxes\n"
        "- A topping is part of its base dish, NOT a separate item "
        "(oil on hummus = one item, spices on rice = one item, "
        "sauce on meat = one item, sesame on bread = one item)\n"
        "- A mixed dish (stew, salad with mixed ingredients) = one item\n"
        "\n"
        '{"items": [\n'
        '  {"name": "kabsa rice", '
        '"description": "golden-yellow rice grains with orange carrot slivers", '
        '"bbox": [120, 80, 450, 350]},\n'
        '  {"name": "grilled chicken", '
        '"description": "charred brown chicken leg with crispy skin", '
        '"bbox": [460, 90, 700, 340]},\n'
        '  {"name": "hummus", '
        '"description": "smooth beige paste with olive oil drizzle on top", '
        '"bbox": [50, 400, 200, 520]},\n'
        '  {"name": "juice box", '
        '"description": "small rectangular cardboard box with colorful label", '
        '"bbox": [710, 30, 790, 180]}\n'
        "]}"
    )
    )


@dataclass
class SAMConfig:
    model_path: str = "facebook/sam3"
    confidence_threshold: float = 0.1
    fallback_thresholds: List[float] = field(default_factory=lambda: [0.05, 0.02, 0.01])
    crop_padding: int = 5
    bpe_search_paths: List[str] = field(default_factory=lambda: [
        "/root/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
    ])


@dataclass
class MatchConfig:
    embedding_model: str = "google/siglip2-base-patch16-224"
    embedding_dim: int = 768
    index_path: str = "menu.index"
    metadata_path: str = "menu_meta.json"
    top_k: int = 3
    similarity_threshold: float = 0.5
    text_weight: float = 0.3          # weight for text signal (0.0 = image only, 1.0 = text only)
    use_text_matching: bool = True    # feature flag to enable/disable hybrid matching


@dataclass
class NMSConfig:
    max_objects: int = 8
    iou_threshold: float = 0.7


@dataclass
class VizConfig:
    draw_boxes: bool = True
    alpha: float = 0.7
    thickness: int = 3
    font_scale: float = 0.9
    font_thickness: int = 2
    show_match_label: bool = True
    show_price: bool = True
    show_confidence: bool = True
    color_low: List[int] = field(default_factory=lambda: [0, 50])
    color_high: List[int] = field(default_factory=lambda: [180, 255])


@dataclass
class PipelineConfig:
    device: str = "cuda"
    output_dir: str = "results"
    sam3_repo_path: str = "sam3"
    image_extensions: List[str] = field(default_factory=lambda: [
        "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp",
    ])
    vlm: VLMConfig = field(default_factory=VLMConfig)
    sam: SAMConfig = field(default_factory=SAMConfig)
    match: MatchConfig = field(default_factory=MatchConfig)
    nms: NMSConfig = field(default_factory=NMSConfig)
    viz: VizConfig = field(default_factory=VizConfig)

    def to_json(self, path: str) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: str) -> "PipelineConfig":
        data = json.loads(Path(path).read_text())
        return cls(
            device=data.get("device", "cuda"),
            output_dir=data.get("output_dir", "results"),
            sam3_repo_path=data.get("sam3_repo_path", "sam3"),
            image_extensions=data.get("image_extensions", ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]),
            vlm=VLMConfig(**data["vlm"]) if "vlm" in data else VLMConfig(),
            sam=SAMConfig(**data["sam"]) if "sam" in data else SAMConfig(),
            match=MatchConfig(**data["match"]) if "match" in data else MatchConfig(),
            nms=NMSConfig(**data["nms"]) if "nms" in data else NMSConfig(),
            viz=VizConfig(**data["viz"]) if "viz" in data else VizConfig(),
        )
