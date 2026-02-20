"""Stage 3: Vector Matching — SigLIP 2 encodes crops → FAISS search → menu item IDs + prices."""

import logging
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor

from config import MatchConfig
from ptypes import SegmentedItem, MatchResult
from vector_store import VectorStore

logger = logging.getLogger("pipeline")


class MenuMatcher:
    """Matches segmented food crops against a reference menu using SigLIP 2 embeddings + FAISS.

    Flow per crop:
    1. Convert BGR crop → PIL RGB
    2. Encode with SigLIP 2 → 768-dim normalized embedding
    3. Query FAISS index → top-k nearest menu items
    4. Apply similarity threshold → "unknown" if below
    """

    def __init__(self, config: MatchConfig, device: str = "cuda"):
        self.config = config
        self.device = device

        # Load SigLIP 2 model
        logger.info(f"Loading embedding model: {config.embedding_model}")
        self.model = AutoModel.from_pretrained(config.embedding_model).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(config.embedding_model)

        # Load FAISS index
        self.store = VectorStore()
        self.store.load(config.index_path, config.metadata_path)
        logger.info(f"MenuMatcher ready: {self.store.index.ntotal} reference vectors")

    def match(self, items: List[SegmentedItem]) -> List[MatchResult]:
        """Match each segmented crop against the menu database.

        Args:
            items: List of SegmentedItem from Stage 2 (with BGR crops).

        Returns:
            List of MatchResult, one per input item.
        """
        results = []
        for item in items:
            result = self._match_single(item)
            results.append(result)
        return results

    def _match_single(self, item: SegmentedItem) -> MatchResult:
        """Match a single crop against the menu."""
        # Encode crop
        embedding = self._encode_crop(item.crop)

        # Query FAISS
        matches = self.store.query(embedding, top_k=self.config.top_k)

        if not matches:
            return MatchResult(
                segmented=item,
                menu_item="unknown",
                category="unknown",
                price=0.0,
                confidence=0.0,
                top_k=[],
            )

        # Build top-k list
        top_k = [(m["name"], score) for m, score in matches]

        # Best match
        best_meta, best_score = matches[0]

        # Apply threshold
        if best_score < self.config.similarity_threshold:
            menu_item = "unknown"
            category = "unknown"
            price = 0.0
        else:
            menu_item = best_meta["name"]
            category = best_meta["category"]
            price = best_meta["price"]

        logger.info(f"Matched '{item.description[:30]}...' → {menu_item} ({best_score:.3f})")

        return MatchResult(
            segmented=item,
            menu_item=menu_item,
            category=category,
            price=price,
            confidence=best_score,
            top_k=top_k,
        )

    def _encode_crop(self, crop: np.ndarray) -> np.ndarray:
        """Encode a BGR crop to a normalized embedding vector.

        Args:
            crop: (H, W, 3) BGR numpy array.

        Returns:
            (dim,) float32 numpy array, L2-normalized.
        """
        # BGR → RGB → PIL
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)

        inputs = self.processor(images=crop_pil, return_tensors="pt").to(self.device)

        with torch.no_grad():
            embedding = self.model.get_image_features(**inputs)
            embedding = F.normalize(embedding, dim=-1)

        return embedding.cpu().numpy().squeeze()
