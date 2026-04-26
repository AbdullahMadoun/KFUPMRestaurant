"""Stage 3 — vector DB classifier.

Drop-in replacement for `FoodClassifier` (the ICL transformer) that does NOT
train. At init, embeds the per-class reference images with a frozen vision
encoder (DINOv2-large by default) and stores per-class prototype vectors. At
inference, embeds the query crop with the same encoder and returns the cosine
top-K class names.

Why this exists
---------------
The TriFoodNet ICL transformer adds ~7-12 pt of accuracy on top of cosine
top-1 retrieval (see HANDOFF.md), but at the cost of joint training, leak
protection, episodic batching, and ~500K trainable params. This module is
the simplest possible alternative: same input/output contract, zero
trainable parameters, no joint training. Use the config flag
`stage3.backend: "vector_db"` in master_config.yaml to wire it in.

Interface
---------
Has the same public surface that `pipeline.run` / `pipeline.run_batched`
need:

    classifier = FoodVectorDB(
        reference_root="data/reference_library",
        class_names=["rice", "grilled fish", ...],
        encoder="facebook/dinov2-large",
        device="cuda",
    )
    preds = classifier.classify(crop_pil_image, device="cuda", top_k=1)
    # → [("rice", 0.84)]
    classifier.last_candidate_class_names  # → ["rice", "grilled fish", ...] top-K names

Eval-harness diagnostic compat
------------------------------
`last_candidate_class_names` is populated identically to the ICL stage so
`evaluate_inference_loop` keeps computing retrieval-vs-transformer
diagnostics — except in this case "transformer" IS retrieval, so the
`stage3_transformer_lift_over_top1` metric will be ~0 by construction.

What this module deliberately does NOT do
-----------------------------------------
- No fine-tuning, no LoRA, no PEFT. The encoder is frozen.
- No episodic sampling. The prototype per class is the *mean* of its
  reference embeddings, computed once at __init__.
- No joint-training integration. `forward()` is unimplemented; this class
  only supports inference. To train, use `stage3_icl.FoodClassifier`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


@dataclass
class VectorDBConfig:
    """All knobs are config-driven so master_config.yaml is the source of truth."""
    encoder: str = "facebook/dinov2-large"
    device: str = "cuda"
    embedding_dim: int = 1024            # DINOv2-large output dim
    use_text_prompt: bool = False        # if True, also embed VLM's name+desc with SigLIP and fuse
    text_encoder: str = "google/siglip2-base-patch16-224"
    text_weight: float = 0.3             # only used when use_text_prompt=True
    refs_per_class_min: int = 1          # gracefully accept classes with few refs
    refs_per_class_max: int = 16         # cap so absurd reference libraries don't blow memory


class FoodVectorDB(nn.Module):
    """Cosine-similarity classifier over per-class reference prototypes."""

    def __init__(
        self,
        reference_root: str | Path,
        class_names: Sequence[str],
        encoder: str = "facebook/dinov2-large",
        device: str = "cuda",
        text_weight: float = 0.0,
        text_encoder: Optional[str] = None,
    ):
        super().__init__()
        self.reference_root = Path(reference_root)
        self.class_names: List[str] = list(class_names)
        self.device_str = device if torch.cuda.is_available() else "cpu"
        self.text_weight = float(text_weight)
        self.use_text = text_weight > 0

        # ── Vision encoder (frozen) ──────────────────────────────────────────
        self.processor = AutoImageProcessor.from_pretrained(encoder)
        self.encoder = AutoModel.from_pretrained(encoder, torch_dtype=torch.float16).to(self.device_str).eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        # ── Optional text encoder for VLM-name retrieval fusion ──────────────
        self.text_processor = None
        self.text_encoder = None
        if self.use_text and text_encoder is not None:
            from transformers import AutoProcessor, AutoModel as TextAutoModel
            self.text_processor = AutoProcessor.from_pretrained(text_encoder)
            self.text_encoder = TextAutoModel.from_pretrained(text_encoder, torch_dtype=torch.float16).to(self.device_str).eval()
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        # ── Build per-class prototypes (one-time) ────────────────────────────
        self.prototypes: torch.Tensor                 # (n_classes, embed_dim), L2-normalized
        self.class_to_idx: Dict[str, int] = {n: i for i, n in enumerate(self.class_names)}
        self._build_prototypes()

        # Diagnostic: filled in by classify(), read by eval_harness.
        self.last_candidate_class_ids: List[int] = []
        self.last_candidate_class_names: List[str] = []

    # ── Prototype construction ───────────────────────────────────────────────

    def _list_reference_images(self) -> Dict[str, List[Path]]:
        """For each class folder under reference_root/, list the reference images."""
        out: Dict[str, List[Path]] = {}
        if not self.reference_root.is_dir():
            raise FileNotFoundError(
                f"reference_root not found: {self.reference_root}. "
                f"Vector DB needs reference/<class_slug>/*.jpg per class."
            )
        for class_name in self.class_names:
            # class folders may be slugs (rice) OR display names (rice). Try both.
            slug = class_name.lower().replace(" ", "_").replace("/", "_")
            for cand in (slug, class_name):
                d = self.reference_root / cand
                if d.is_dir():
                    images = sorted([p for p in d.iterdir()
                                     if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")])
                    out[class_name] = images
                    break
            else:
                out[class_name] = []
        return out

    @torch.inference_mode()
    def _embed_image(self, pil_image: Image.Image) -> torch.Tensor:
        """Returns one (embed_dim,) L2-normalized vector for one image."""
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device_str)
        out = self.encoder(**inputs)
        # DINOv2-style: CLS token from last_hidden_state
        emb = out.last_hidden_state[:, 0]            # (1, embed_dim)
        emb = F.normalize(emb.float(), dim=-1)       # cast back to fp32 for stable norm
        return emb.squeeze(0).cpu()

    def _build_prototypes(self):
        """Embed every reference image, mean-pool per class, L2-normalize."""
        ref_paths = self._list_reference_images()
        n_classes = len(self.class_names)
        protos: List[Optional[torch.Tensor]] = [None] * n_classes

        for class_name, paths in ref_paths.items():
            cid = self.class_to_idx[class_name]
            if not paths:
                # No refs at all — placeholder zero vector. Cosine to it is 0.
                protos[cid] = torch.zeros(self._embedding_dim_probe())
                continue
            embs = []
            for p in paths:
                try:
                    with Image.open(p) as img:
                        embs.append(self._embed_image(img.convert("RGB")))
                except Exception as e:
                    print(f"[FoodVectorDB] skipping bad ref {p}: {e}")
            if not embs:
                protos[cid] = torch.zeros(self._embedding_dim_probe())
                continue
            mean_emb = torch.stack(embs).mean(dim=0)
            mean_emb = F.normalize(mean_emb, dim=-1)
            protos[cid] = mean_emb

        self.prototypes = torch.stack(protos).to(self.device_str)   # (n_classes, embed_dim)
        # Re-normalize after stacking (paranoia)
        self.prototypes = F.normalize(self.prototypes, dim=-1)

    def _embedding_dim_probe(self) -> int:
        """Heuristic for placeholder zero-vector size, used only for empty-ref classes."""
        if hasattr(self.encoder.config, "hidden_size"):
            return int(self.encoder.config.hidden_size)
        # DINOv2-large fallback
        return 1024

    # ── Inference ────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def classify(
        self,
        query_image: Image.Image,
        device: str = "cuda",
        top_k: int = 1,
    ) -> List[Tuple[str, float]]:
        """Return list of (class_name, cosine_score) sorted descending."""
        if not isinstance(query_image, Image.Image):
            # Allow tensor input for compatibility with the joint pipeline
            from torchvision.transforms.functional import to_pil_image
            query_image = to_pil_image(query_image)

        query_emb = self._embed_image(query_image.convert("RGB")).to(self.device_str)
        # cosine since both are L2-normalized
        scores = self.prototypes @ query_emb.unsqueeze(-1)  # (n_classes, 1)
        scores = scores.squeeze(-1)                          # (n_classes,)

        k = min(max(1, int(top_k)), self.prototypes.shape[0])
        top_scores, top_indices = scores.topk(k=k)

        # Diagnostic — set the same fields as stage3_icl so eval_harness's
        # candidate_classes-based metrics keep working.
        self.last_candidate_class_ids = top_indices.tolist()
        self.last_candidate_class_names = [self.class_names[i] for i in self.last_candidate_class_ids]

        return [
            (self.class_names[int(idx)], float(s))
            for s, idx in zip(top_scores.tolist(), top_indices.tolist())
        ]

    # ── Training surface (intentionally unimplemented) ───────────────────────

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "FoodVectorDB is inference-only. To train Stage 3, swap "
            "`master_config.yaml::stage3.backend` back to `icl_transformer` "
            "and use stage3_icl.FoodClassifier."
        )

    def set_support_set(self, *args, **kwargs):
        # No-op: prototypes are already built from reference_root at __init__.
        # We accept the call so callers expecting the ICL interface don't crash.
        return None
