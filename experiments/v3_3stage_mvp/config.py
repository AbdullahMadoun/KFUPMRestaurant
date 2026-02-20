"""Pipeline configuration as nested stdlib dataclasses with JSON I/O."""

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
from typing import List


@dataclass
class VLMConfig:
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    gpu_memory_utilization: float = 0.6
    max_model_len: int = 2048
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
        "For EACH portion, describe ONLY what your eyes see:\n"
        "- Main color(s) and any color variations\n"
        "- Surface texture (glossy, matte, grainy, smooth, rough, flaky)\n"
        "- Shape and form (mound, flat spread, scattered pieces, slices, chunks, strips)\n"
        "- Position on plate (left, right, center, top, bottom)\n"
        "- Relative size (fraction of plate area)\n"
        "\n"
        "GROUPING RULES:\n"
        "- Multiple scattered pieces of the SAME food = ONE item with ONE bbox around all of them\n"
        "  (e.g. several small chunks spread across a region = single item, single bbox)\n"
        "- A mixed dish (salad, stew, stir-fry with mixed ingredients) = ONE item\n"
        "- Visually DIFFERENT foods that touch or overlap = SEPARATE items\n"
        "- Same food in clearly separated areas with a visible gap = SEPARATE items\n"
        "- Small garnishes (lemon slice, herbs) = separate item only if clearly visible\n"
        "\n"
        "Count carefully. Cafeteria plates typically have 1-5 portions but "
        "do NOT default to any specific number. Count what you actually see.\n"
        "\n"
        "Return strictly as JSON:\n"
        '{"items": [\n'
        '  {"description": "<main_colors> <surface_texture> <shape_and_form>, '
        '<position_on_plate>", "bbox": [x1, y1, x2, y2]},\n'
        "  ...\n"
        "]}\n"
        "\n"
        "The description template above shows the FORMAT — you MUST replace every "
        "<placeholder> with actual visual details from THIS specific image. "
        "Never output angle brackets or placeholder text.\n"
        "Bounding boxes are pixel coordinates [x1, y1, x2, y2] tightly fitting each item.\n"
        "Return ONLY the JSON, no other text."
    )


@dataclass
class SAMConfig:
    model_path: str = "facebook/sam3"
    confidence_threshold: float = 0.15
    fallback_thresholds: List[float] = field(default_factory=lambda: [0.05, 0.02, 0.01])
    crop_padding: int = 10
    bpe_search_paths: List[str] = field(default_factory=lambda: [
        "/root/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
    ])
    multi_box_prompt: bool = True     # send 2x2 sub-box grid + full bbox (5 total)
    multi_box_grid: int = 2           # NxN sub-box grid (2 = 4 sub-boxes + 1 full = 5 prompts)


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
