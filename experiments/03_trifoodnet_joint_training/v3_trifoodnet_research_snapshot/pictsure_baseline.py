# =============================================================================
# FILE: pictsure_baseline.py
# CATEGORY: ARCH
# PURPOSE: Local baseline implementation used for ablations against official PictSure.
# DEPENDENCIES: None
# USED BY: None
# KEY CLASSES/FUNCTIONS: CLIPEncoder, PictSureIndex, PictSureClassifier, PictSurePipeline, _softmax
# LAST MODIFIED: 2026-03-21T07:08:22+00:00
# SNAPSHOT NOTES: no major inline issues detected during snapshot generation
# =============================================================================
"""
trifoodnet/models/pictsure_baseline.py
──────────────────────────────────────
PictSure baseline — local CLIP retrieval implementation used in V1 of this
project.

This file intentionally keeps the pre-official baseline implementation for
ablation work. The actual pretrained PictSure models from Hugging Face are
integrated separately via `pictsure_official.py` so this module no longer
shadows the upstream `PictSure` package on Windows.

The baseline is an image-level retrieval system:
  1. Encode a query image with a visual backbone (CLIP by default).
  2. Compute cosine similarity against a pre-built reference embedding index.
  3. Return the top-k most similar items from the index.

This module provides:
  • PictSureIndex   — builds and queries the embedding index
  • PictSureClassifier — wraps index into a classifier interface
  • PictSurePipeline — full pipeline: raw image → crop → classify

References
──────────
PictSure was originally described as a plug-in visual search engine for
unstructured product catalogues.  This re-implementation follows the same
two-stage pattern: (a) visual embedding via CLIP, (b) nearest-neighbour search.
We extend it with:
  • HNSW-accelerated ANN search via `hnswlib`
  • Confidence calibration via temperature scaling
  • Optional L2/cosine dual-metric scoring
  • Save/load serialization for production deployment
"""

from __future__ import annotations
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

try:
    import hnswlib
    HAS_HNSWLIB = True
except ImportError:
    HAS_HNSWLIB = False


# ──────────────────────────────────────────────────────────────────────────────
# Encoder
# ──────────────────────────────────────────────────────────────────────────────

class CLIPEncoder:
    """
    Thin stateless wrapper around CLIP for embedding extraction.
    All methods are `@torch.no_grad()` — this encoder is never trained.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: str     = "cuda",
        batch_size: int = 64,
    ):
        self.device     = device
        self.batch_size = batch_size
        self.model      = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor  = CLIPProcessor.from_pretrained(model_name)
        self.embed_dim  = self.model.config.projection_dim

        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, images: List[Image.Image]) -> np.ndarray:
        """Encode a list of PIL images → float32 numpy [N, D] (L2-normalised)."""
        all_embeds = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            feats  = self.model.get_image_features(**inputs)
            feats  = F.normalize(feats, dim=-1)
            all_embeds.append(feats.cpu().float().numpy())
        return np.vstack(all_embeds)

    @torch.no_grad()
    def encode_tensor(self, tensor: torch.Tensor) -> np.ndarray:
        """Encode a pre-processed [N, 3, 224, 224] tensor (CLIP normalisation)."""
        all_embeds = []
        for i in range(0, len(tensor), self.batch_size):
            batch = tensor[i:i + self.batch_size].to(self.device)
            feats = self.model.get_image_features(pixel_values=batch)
            feats = F.normalize(feats, dim=-1)
            all_embeds.append(feats.cpu().float().numpy())
        return np.vstack(all_embeds)


# ──────────────────────────────────────────────────────────────────────────────
# Index
# ──────────────────────────────────────────────────────────────────────────────

class PictSureIndex:
    """
    Embedding index supporting:
      • Exact brute-force cosine search (default, always available)
      • Approximate nearest-neighbour via HNSW (requires hnswlib)

    Parameters
    ----------
    embed_dim   : dimensionality of embeddings (512 for CLIP ViT-L/14)
    use_hnsw    : use HNSW for ANN search (faster for large indices)
    hnsw_ef     : HNSW search parameter (higher = more accurate, slower)
    hnsw_M      : HNSW construction parameter (higher = more memory, more accurate)
    temperature : softmax temperature for confidence scores
    """

    def __init__(
        self,
        embed_dim:   int   = 512,
        use_hnsw:    bool  = False,
        hnsw_ef:     int   = 200,
        hnsw_M:      int   = 32,
        temperature: float = 0.07,
    ):
        self.embed_dim   = embed_dim
        self.use_hnsw    = use_hnsw and HAS_HNSWLIB
        self.hnsw_ef     = hnsw_ef
        self.hnsw_M      = hnsw_M
        self.temperature = temperature

        # Storage
        self.embeddings:   np.ndarray = np.zeros((0, embed_dim), dtype=np.float32)
        self.labels:       List[str]  = []           # class name per embedding
        self.label_ids:    List[int]  = []
        self.label2id:     Dict[str, int] = {}
        self.id2label:     Dict[int, str] = {}

        self._hnsw_index = None
        self._dirty      = False

    # ── building the index ────────────────────────────────────────────────────

    def add(self, embeddings: np.ndarray, class_names: List[str]):
        """
        Add embeddings with corresponding class names.
        embeddings : float32 [N, D] (already L2-normalised)
        class_names: list of N strings
        """
        assert len(embeddings) == len(class_names)

        for name in class_names:
            if name not in self.label2id:
                new_id = len(self.label2id)
                self.label2id[name] = new_id
                self.id2label[new_id] = name

        ids = [self.label2id[n] for n in class_names]

        self.embeddings = np.vstack([self.embeddings, embeddings]) \
            if len(self.embeddings) > 0 else embeddings.copy()
        self.labels    += class_names
        self.label_ids += ids
        self._dirty = True

    def build(self):
        """(Re-)build HNSW index after adding embeddings."""
        if not self.use_hnsw:
            return
        n = len(self.embeddings)
        if n == 0:
            return
        self._hnsw_index = hnswlib.Index(space="cosine", dim=self.embed_dim)
        self._hnsw_index.init_index(max_elements=n, ef_construction=200, M=self.hnsw_M)
        self._hnsw_index.add_items(self.embeddings, list(range(n)))
        self._hnsw_index.set_ef(self.hnsw_ef)
        self._dirty = False
        print(f"[PictSureIndex] HNSW index built with {n} embeddings.")

    # ── querying ──────────────────────────────────────────────────────────────

    def query(
        self,
        query_embeds: np.ndarray,       # [Q, D] L2-normalised
        top_k: int = 5,
    ) -> List[List[Tuple[str, float]]]:
        """
        Returns list of Q result lists.  Each result list contains
        (class_name, confidence) tuples sorted by confidence descending.
        """
        if self._dirty and self.use_hnsw:
            self.build()

        if self.use_hnsw and self._hnsw_index is not None:
            return self._query_hnsw(query_embeds, top_k)
        return self._query_exact(query_embeds, top_k)

    def _query_exact(
        self, query_embeds: np.ndarray, top_k: int
    ) -> List[List[Tuple[str, float]]]:
        """Brute-force cosine similarity search."""
        sims = query_embeds @ self.embeddings.T             # [Q, N_total]
        results = []
        for row in sims:
            idx = np.argsort(-row)[:top_k]
            # Aggregate by class: mean similarity per class among top-k
            class_sims: Dict[str, List[float]] = {}
            for i in idx:
                name = self.labels[i]
                class_sims.setdefault(name, []).append(float(row[i]))
            # Mean sim per class → softmax confidence
            class_mean = {n: np.mean(v) for n, v in class_sims.items()}
            scores = np.array(list(class_mean.values())) / self.temperature
            probs  = _softmax(scores)
            results.append(
                sorted(zip(class_mean.keys(), probs.tolist()), key=lambda x: -x[1])
            )
        return results

    def _query_hnsw(
        self, query_embeds: np.ndarray, top_k: int
    ) -> List[List[Tuple[str, float]]]:
        labels_out, dists_out = self._hnsw_index.knn_query(
            query_embeds, k=min(top_k * 3, len(self.embeddings))
        )
        results = []
        for indices, distances in zip(labels_out, dists_out):
            sims = 1.0 - distances                       # cosine: 1 - distance
            class_sims: Dict[str, List[float]] = {}
            for i, s in zip(indices, sims):
                name = self.labels[i]
                class_sims.setdefault(name, []).append(float(s))
            class_mean = {n: np.mean(v) for n, v in class_sims.items()}
            scores = np.array(list(class_mean.values())) / self.temperature
            probs  = _softmax(scores)
            results.append(
                sorted(zip(class_mean.keys(), probs.tolist()), key=lambda x: -x[1])[:top_k]
            )
        return results

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "embeddings.npy", self.embeddings)
        meta = {
            "labels":    self.labels,
            "label_ids": self.label_ids,
            "label2id":  self.label2id,
            "id2label":  {int(k): v for k, v in self.id2label.items()},
            "embed_dim": self.embed_dim,
            "temperature": self.temperature,
        }
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        if self._hnsw_index is not None:
            self._hnsw_index.save_index(str(path / "hnsw.bin"))
        print(f"[PictSureIndex] Saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "PictSureIndex":
        path = Path(path)
        with open(path / "meta.json") as f:
            meta = json.load(f)
        idx = cls(embed_dim=meta["embed_dim"], temperature=meta["temperature"])
        idx.embeddings = np.load(path / "embeddings.npy")
        idx.labels     = meta["labels"]
        idx.label_ids  = meta["label_ids"]
        idx.label2id   = meta["label2id"]
        idx.id2label   = {int(k): v for k, v in meta["id2label"].items()}
        hnsw_path = path / "hnsw.bin"
        if hnsw_path.exists() and HAS_HNSWLIB:
            idx.use_hnsw = True
            idx._hnsw_index = hnswlib.Index(space="cosine", dim=idx.embed_dim)
            idx._hnsw_index.load_index(str(hnsw_path))
        print(f"[PictSureIndex] Loaded {len(idx.embeddings)} embeddings from {path}")
        return idx

    @property
    def num_classes(self) -> int:
        return len(self.label2id)

    @property
    def num_embeddings(self) -> int:
        return len(self.embeddings)


# ──────────────────────────────────────────────────────────────────────────────
# Classifier interface
# ──────────────────────────────────────────────────────────────────────────────

class PictSureClassifier:
    """
    High-level classifier interface over PictSureIndex + CLIPEncoder.

    Parameters
    ----------
    encoder      : CLIPEncoder instance
    index        : PictSureIndex instance (can be loaded from disk)
    top_k        : how many class candidates to return
    """

    def __init__(
        self,
        encoder: CLIPEncoder,
        index:   Optional[PictSureIndex] = None,
        top_k:   int = 5,
    ):
        self.encoder = encoder
        self.index   = index or PictSureIndex(embed_dim=encoder.embed_dim)
        self.top_k   = top_k

    def add_reference_images(
        self,
        class_name: str,
        images:     List[Image.Image],
    ):
        """Register reference images for a class.  No retraining required."""
        embeddings = self.encoder.encode(images)
        self.index.add(embeddings, [class_name] * len(images))

    def build(self):
        """Build ANN index after adding all references."""
        self.index.build()

    def classify(
        self,
        images: List[Image.Image],
        top_k:  Optional[int] = None,
    ) -> List[List[Tuple[str, float]]]:
        """
        Classify a list of PIL images.
        Returns list of [(class_name, confidence)] per image.
        """
        k = top_k or self.top_k
        embeddings = self.encoder.encode(images)
        return self.index.query(embeddings, top_k=k)

    def classify_tensor(
        self,
        tensors: torch.Tensor,           # [N, 3, 224, 224] CLIP-normalised
        top_k:   Optional[int] = None,
    ) -> List[List[Tuple[str, float]]]:
        k = top_k or self.top_k
        embeddings = self.encoder.encode_tensor(tensors)
        return self.index.query(embeddings, top_k=k)

    def save(self, path: str | Path):
        self.index.save(path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cuda") -> "PictSureClassifier":
        index   = PictSureIndex.load(path)
        encoder = CLIPEncoder(device=device)
        if encoder.embed_dim != index.embed_dim:
            raise ValueError(
                f"Loaded index expects embed_dim={index.embed_dim}, "
                f"but encoder provides {encoder.embed_dim}. "
                "Rebuild the index with the requested backbone."
            )
        return cls(encoder, index)


# ──────────────────────────────────────────────────────────────────────────────
# Full PictSure pipeline (image → crops → classify)
# ──────────────────────────────────────────────────────────────────────────────

class PictSurePipeline:
    """
    Standalone V1-style PictSure pipeline:
      raw dish image → box crops (from any detector) → PictSure classification

    This can be used as a baseline (Table 2) against the ICL Transformer.

    Parameters
    ----------
    classifier : PictSureClassifier
    crop_size  : resize all crops to this before CLIP encoding
    """

    def __init__(
        self,
        classifier: PictSureClassifier,
        crop_size:  int = 224,
    ):
        self.classifier = classifier
        self.crop_size  = crop_size

    def run(
        self,
        image:     Image.Image,
        boxes:     List[List[float]],    # [[x1,y1,x2,y2], ...]
        labels:    Optional[List[str]]   = None,
        top_k:     int                   = 1,
    ) -> List[Dict]:
        """
        Parameters
        ----------
        image  : full dish PIL image
        boxes  : bounding boxes (in pixel coords) from Stage 1
        labels : optional coarse labels from Stage 1 (ignored for ranking)
        top_k  : number of candidate classes per crop

        Returns
        -------
        list of {box, predictions: [(class_name, confidence)]}
        """
        crops = []
        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            crop = image.crop((x1, y1, x2, y2)).resize(
                (self.crop_size, self.crop_size), Image.BICUBIC
            )
            crops.append(crop)

        if not crops:
            return []

        predictions = self.classifier.classify(crops, top_k=top_k)

        results = []
        for box, preds in zip(boxes, predictions):
            results.append({
                "box":         box,
                "coarse_label": labels[len(results)] if labels else None,
                "predictions": preds,     # [(class_name, confidence)]
                "best_class":  preds[0][0] if preds else None,
                "confidence":  preds[0][1] if preds else 0.0,
            })

        return results


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()
