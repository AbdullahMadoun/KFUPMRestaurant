"""Pipeline configuration as nested stdlib dataclasses with JSON I/O."""

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
from typing import List


@dataclass
class VLMConfig:
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    gpu_memory_utilization: float = 0.4
    max_model_len: int = 4096
    enforce_eager: bool = True
    allowed_local_media_path: str = "/root"
    temperature: float = 0.1
    max_tokens: int = 512
    describe_template: str = (
        "Look at this plate of food. For each distinct food portion visible:\n"
        "\n"
        "1. Describe its VISUAL APPEARANCE only: color, texture, shape, surface pattern\n"
        "2. Provide its bounding box as [x1, y1, x2, y2] in pixel coordinates\n"
        "\n"
        "RULES:\n"
        "- Describe what you SEE, not what you think it is called\n"
        "- A mixed dish (salad, stew) = ONE item — do NOT split ingredients\n"
        "- Side-by-side items on same plate = SEPARATE items\n"
        "- Ignore plates, bowls, cutlery, wrapping, background\n"
        "\n"
        'Return strictly as JSON:\n'
        '{"items": [\n'
        '  {"description": "golden brown crispy surface with ridged texture, roughly oval shaped", "bbox": [x1, y1, x2, y2]},\n'
        '  ...\n'
        ']}\n'
        "Return ONLY the JSON."
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
