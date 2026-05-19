# =============================================================================
# FILE: pictsure_official.py
# CATEGORY: ARCH
# PURPOSE: Adapter layer over the upstream public PictSure checkpoints.
# DEPENDENCIES: None
# USED BY: run_official_pictsure.py
# KEY CLASSES/FUNCTIONS: resolve_device, resolve_autocast_dtype, load_reference_library_from_dir, OfficialPictSureClassifier
# LAST MODIFIED: 2026-03-21T07:12:02+00:00
# SNAPSHOT NOTES: no major inline issues detected during snapshot generation
# =============================================================================
"""
Official PictSure integration for public Hugging Face checkpoints.

This adapter wraps the upstream `PictSure` package so the rest of this
repository can use the same `build_reference_library()` / `classify()` style
interface as the local Stage 3 classifier.
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image


Prediction = List[Tuple[str, float]]


def resolve_device(requested: Optional[str | torch.device] = None) -> torch.device:
    if isinstance(requested, torch.device):
        return requested
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_autocast_dtype(requested: str | torch.dtype) -> torch.dtype:
    if isinstance(requested, torch.dtype):
        return requested
    name = requested.lower().strip()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported autocast dtype: {requested}")
    return mapping[name]


def load_reference_library_from_dir(root: str | Path) -> Dict[str, List[Image.Image]]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Reference library path not found: {root_path}")

    library: Dict[str, List[Image.Image]] = {}
    for class_dir in sorted(p for p in root_path.iterdir() if p.is_dir()):
        images: List[Image.Image] = []
        for img_path in sorted(class_dir.glob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            with Image.open(img_path) as image:
                images.append(image.convert("RGB").copy())
        if images:
            library[class_dir.name] = images

    if not library:
        raise ValueError(f"No reference images found under {root_path}")
    return library


class OfficialPictSureClassifier:
    """
    Wrapper around the upstream PictSure package.

    The public upstream API is:
      model = PictSure.from_pretrained("pictsure/pictsure-vit")
      model.set_context_images(reference_images, reference_labels)
      predictions = model.predict(query_images)

    This class keeps that flow but normalizes outputs to:
      [("class_name", confidence), ...]
    """

    def __init__(
        self,
        model_id: str = "pictsure/pictsure-vit",
        device: Optional[str | torch.device] = "auto",
        use_torch_compile: bool = False,
        compile_mode: str = "default",
        use_autocast: bool = True,
        autocast_dtype: str | torch.dtype = torch.float16,
        hf_token_env: str = "HF_TOKEN",
    ):
        self.model_id = model_id
        self.device = resolve_device(device)
        self.use_torch_compile = use_torch_compile
        self.compile_mode = compile_mode
        self.use_autocast = use_autocast
        self.autocast_dtype = resolve_autocast_dtype(autocast_dtype)
        self.hf_token_env = hf_token_env

        self.model: Any = None
        self._context_images: List[Image.Image] = []
        self._context_labels: List[str] = []

    def _ensure_model(self):
        if self.model is not None:
            return self.model

        token = os.environ.get(self.hf_token_env)
        if token:
            os.environ.setdefault("HF_TOKEN", token)

        try:
            from PictSure import PictSure as UpstreamPictSure
        except ImportError as exc:
            raise ImportError(
                "Official PictSure package is not installed. "
                "Install it with `pip install PictSure`."
            ) from exc

        model = UpstreamPictSure.from_pretrained(self.model_id)
        if hasattr(model, "to"):
            model = model.to(self.device)
        if isinstance(model, torch.nn.Module):
            model.eval()
            if self.use_torch_compile and hasattr(torch, "compile"):
                try:
                    model = torch.compile(model, mode=self.compile_mode)
                except Exception as exc:
                    print(f"[OfficialPictSureClassifier] torch.compile skipped: {exc}")

        self.model = model
        return self.model

    def build_reference_library(
        self,
        class_images: Dict[str, Sequence[Image.Image]],
        device: Optional[str | torch.device] = None,
    ):
        model = self._ensure_model()
        self.device = resolve_device(device or self.device)

        flat_images: List[Image.Image] = []
        flat_labels: List[str] = []
        for class_name, images in class_images.items():
            if not images:
                continue
            flat_images.extend(images)
            flat_labels.extend([class_name] * len(images))

        if not flat_images:
            raise ValueError("Reference library is empty.")

        if hasattr(model, "to"):
            model = model.to(self.device)
            self.model = model

        with torch.inference_mode():
            if hasattr(model, "set_context_images"):
                model.set_context_images(flat_images, flat_labels)
            elif hasattr(model, "set_context"):
                model.set_context(flat_images, flat_labels)
            else:
                raise AttributeError(
                    "Official PictSure model has no `set_context_images` or "
                    "`set_context` method."
                )

        self._context_images = list(flat_images)
        self._context_labels = list(flat_labels)

    def add_class(
        self,
        class_name: str,
        pil_images: Sequence[Image.Image],
        device: Optional[str | torch.device] = None,
    ):
        if not pil_images:
            raise ValueError("`pil_images` must contain at least one image.")
        self._context_images.extend(pil_images)
        self._context_labels.extend([class_name] * len(pil_images))
        grouped: Dict[str, List[Image.Image]] = {}
        for image, label in zip(self._context_images, self._context_labels):
            grouped.setdefault(label, []).append(image)
        self.build_reference_library(grouped, device=device)

    @torch.inference_mode()
    def classify(
        self,
        crop_image: Image.Image,
        device: Optional[str | torch.device] = None,
        top_k: int = 1,
    ) -> Prediction:
        if not self._context_labels:
            raise RuntimeError("Call build_reference_library() before classify().")

        model = self._ensure_model()
        self.device = resolve_device(device or self.device)
        if hasattr(model, "to"):
            model = model.to(self.device)
            self.model = model

        use_cuda_autocast = (
            self.use_autocast
            and self.device.type == "cuda"
            and self.autocast_dtype in {torch.float16, torch.bfloat16}
        )
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=self.autocast_dtype)
            if use_cuda_autocast
            else nullcontext()
        )

        with autocast_ctx:
            if not hasattr(model, "predict"):
                raise AttributeError("Official PictSure model has no `predict` method.")
            raw_predictions = model.predict([crop_image])

        predictions = self._normalize_predictions(raw_predictions)
        return predictions[:max(1, top_k)]

    def _normalize_predictions(self, raw_predictions: Any) -> Prediction:
        if isinstance(raw_predictions, list):
            if not raw_predictions:
                return []
            first = raw_predictions[0]
            if isinstance(first, str):
                return [(first, 1.0)]
            if isinstance(first, tuple) and len(first) == 2:
                return self._normalize_ranked_list(raw_predictions)
            if isinstance(first, dict):
                return self._normalize_ranked_dicts(raw_predictions)
            if isinstance(first, list):
                return self._normalize_predictions(first)

        if isinstance(raw_predictions, tuple) and len(raw_predictions) == 2:
            labels, scores = raw_predictions
            if isinstance(labels, Iterable) and isinstance(scores, Iterable):
                return [
                    (str(label), float(score))
                    for label, score in zip(labels, scores)
                ]

        if isinstance(raw_predictions, dict):
            if "predictions" in raw_predictions:
                return self._normalize_predictions(raw_predictions["predictions"])
            label = raw_predictions.get("label") or raw_predictions.get("class_name")
            if label is not None:
                score = raw_predictions.get("score", raw_predictions.get("confidence", 1.0))
                return [(str(label), float(score))]

        if isinstance(raw_predictions, str):
            return [(raw_predictions, 1.0)]

        raise TypeError(
            "Unsupported prediction format returned by official PictSure: "
            f"{type(raw_predictions)!r}"
        )

    @staticmethod
    def _normalize_ranked_list(values: Sequence[Tuple[Any, Any]]) -> Prediction:
        return [(str(label), float(score)) for label, score in values]

    @staticmethod
    def _normalize_ranked_dicts(values: Sequence[Dict[str, Any]]) -> Prediction:
        normalized: Prediction = []
        for item in values:
            label = item.get("label") or item.get("class_name") or item.get("prediction")
            if label is None:
                continue
            score = item.get("score", item.get("confidence", item.get("probability", 1.0)))
            normalized.append((str(label), float(score)))
        return normalized
