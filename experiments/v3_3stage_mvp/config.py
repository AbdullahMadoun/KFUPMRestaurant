"""Pipeline configuration as nested stdlib dataclasses with JSON I/O."""

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
from typing import List


@dataclass
class VLMConfig:
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 4096
    enforce_eager: bool = False
    enable_prefix_caching: bool = True
    allowed_local_media_path: str = "/root"
    temperature: float = 0.3
    max_tokens: int = 768
    top_p: float = 0.8
    top_k: int = 20
    system_prompt: str = (
        "You are a food visual analyzer. You describe exactly what you see "
        "in images — colors, textures, shapes, positions. You never guess "
        "food names. You output only valid JSON."
    )
    describe_template: str = (
        "Look at this cafeteria plate and identify every distinct food portion.\n"
        "\n"
        "For EACH portion provide:\n"
        "- description: short visual description (~10 words) — colors, texture, shape\n"
        "- bbox: tight bounding box [x1, y1, x2, y2] in pixel coordinates\n"
        "- points: 2-3 pixel coordinates [x, y] placed directly ON the food surface. "
        "Pick points that are clearly on the food, spread across the region.\n"
        "\n"
        "GROUPING RULES:\n"
        "- Scattered pieces of the SAME food = ONE item, ONE bbox around ALL pieces\n"
        "- A mixed dish (salad, stew, stir-fry) = ONE item\n"
        "- Visually DIFFERENT foods that touch = SEPARATE items\n"
        "- Same food in clearly separated areas = SEPARATE items\n"
        "\n"
        "Count what you actually see. Do NOT default to any number.\n"
        "\n"
        "Return strictly as JSON:\n"
        '{"items": [\n'
        '  {"description": "<colors> <texture> <shape>", '
        '"bbox": [x1, y1, x2, y2], '
        '"points": [[px1, py1], [px2, py2]]},\n'
        "  ...\n"
        "]}\n"
        "\n"
        "Replace <placeholders> with actual observations. "
        "All coordinates are pixel values. Return ONLY the JSON."
    )


@dataclass
class SAMConfig:
    model_path: str = "facebook/sam3"
    confidence_threshold: float = 0.1
    fallback_thresholds: List[float] = field(default_factory=lambda: [0.05, 0.02, 0.01])
    crop_padding: int = 10
    bpe_search_paths: List[str] = field(default_factory=lambda: [
        "/root/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
    ])
    multi_box_prompt: bool = False    # set True to send 2x2 sub-box grid + full bbox (5 total)
    multi_box_grid: int = 2           # NxN sub-box grid (2 = 4 sub-boxes + 1 full = 5 prompts)
    use_vlm_points: bool = True       # use VLM-provided foreground points (from Stage 1)


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
