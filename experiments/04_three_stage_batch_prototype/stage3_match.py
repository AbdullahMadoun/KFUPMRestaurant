"""Stage 3: Vector Matching — SigLIP 2 encodes crops → FAISS search → menu item IDs + prices.

Supports hybrid text+image matching via dual-query late fusion:
  1. Image query: encode crop → FAISS search
  2. Text query: encode VLM description → FAISS search
  3. Aggregate scores per menu item: final = (1-w)*img + w*txt
"""

import logging
from typing import Dict, List, Tuple

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
        self.model = AutoModel.from_pretrained(config.embedding_model, torch_dtype=torch.float16).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(config.embedding_model)

        # Load FAISS index
        self.store = VectorStore()
        self.store.load(config.index_path, config.metadata_path)
        logger.info(f"MenuMatcher ready: {self.store.index.ntotal} reference vectors")

    def match(self, items: List[SegmentedItem]) -> List[MatchResult]:
        """Match segmented crops against the menu database using batched encoding.

        Args:
            items: List of SegmentedItem from Stage 2 (with BGR crops).

        Returns:
            List of MatchResult, one per input item.
        """
        if not items:
            return []

        # Batch encode all crops (1 forward pass instead of N)
        crop_pils = [Image.fromarray(cv2.cvtColor(it.crop, cv2.COLOR_BGR2RGB)) for it in items]
        image_embs = self._encode_images_batch(crop_pils)

        # Batch encode all text descriptions (if hybrid enabled)
        text_embs = None
        if self.config.use_text_matching:
            descriptions = [it.description for it in items]
            if any(descriptions):
                text_embs = self._encode_texts_batch(descriptions)

        # Batch FAISS queries
        image_results = self.store.query_batch(image_embs, top_k=self.config.top_k)
        text_results = None
        if text_embs is not None:
            text_results = self.store.query_batch(text_embs, top_k=self.config.top_k)

        # Aggregate per item (CPU, fast)
        results = []
        for i, item in enumerate(items):
            img_matches = image_results[i]
            txt_matches = text_results[i] if text_results else []
            result = self._build_result(item, img_matches, txt_matches)
            results.append(result)
        return results

    def _build_result(
        self, item: SegmentedItem,
        image_matches: List[Tuple[dict, float]],
        text_matches: List[Tuple[dict, float]],
    ) -> MatchResult:
        """Build a MatchResult from pre-computed FAISS matches."""
        aggregated = self._aggregate_scores(image_matches, text_matches)

        if not aggregated:
            return MatchResult(
                segmented=item,
                menu_item="unknown",
                category="unknown",
                price=0.0,
                confidence=0.0,
                top_k=[],
            )

        # Build top-k list from aggregated results
        top_k = [(meta["name"], final) for meta, final, _, _ in aggregated]

        # Best match
        best_meta, best_score, best_img, best_txt = aggregated[0]

        # Apply threshold
        if best_score < self.config.similarity_threshold:
            menu_item = "unknown"
            category = "unknown"
            price = 0.0
        else:
            menu_item = best_meta["name"]
            category = best_meta["category"]
            price = best_meta["price"]

        logger.info(
            f"Matched '{item.description[:30]}...' -> {menu_item} "
            f"(final={best_score:.3f}, img={best_img:.3f}, txt={best_txt:.3f})"
        )

        return MatchResult(
            segmented=item,
            menu_item=menu_item,
            category=category,
            price=price,
            confidence=best_score,
            top_k=top_k,
            image_score=best_img,
            text_score=best_txt,
        )

    def _match_single(self, item: SegmentedItem) -> MatchResult:
        """Match a single crop against the menu using dual query (image + text).

        Kept for backward compatibility / single-query use cases.
        """
        image_emb = self._encode_crop(item.crop)
        image_matches = self.store.query(image_emb, top_k=self.config.top_k)

        if self.config.use_text_matching and item.description:
            text_emb = self._encode_text(item.description)
            text_matches = self.store.query(text_emb, top_k=self.config.top_k)
        else:
            text_matches = []

        return self._build_result(item, image_matches, text_matches)

    def _aggregate_scores(
        self,
        image_matches: List[Tuple[dict, float]],
        text_matches: List[Tuple[dict, float]],
    ) -> List[Tuple[dict, float, float, float]]:
        """Merge image and text FAISS results by item name.

        Returns list of (metadata, final_score, image_score, text_score) sorted by final_score desc.
        """
        w = self.config.text_weight

        # Collect best score per item name from each signal
        image_by_name: Dict[str, Tuple[dict, float]] = {}
        text_by_name: Dict[str, Tuple[dict, float]] = {}

        for meta, score in image_matches:
            name = meta["name"]
            if name not in image_by_name or score > image_by_name[name][1]:
                image_by_name[name] = (meta, score)

        for meta, score in text_matches:
            name = meta["name"]
            if name not in text_by_name or score > text_by_name[name][1]:
                text_by_name[name] = (meta, score)

        # Union of all item names seen
        all_names = set(image_by_name.keys()) | set(text_by_name.keys())

        results = []
        for name in all_names:
            img_meta, img_score = image_by_name.get(name, (None, 0.0))
            txt_meta, txt_score = text_by_name.get(name, (None, 0.0))
            meta = img_meta or txt_meta  # prefer image metadata (has source_image)

            final_score = (1 - w) * img_score + w * txt_score
            results.append((meta, final_score, img_score, txt_score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _encode_images_batch(self, images: List[Image.Image]) -> np.ndarray:
        """Batch encode PIL images to normalized embeddings.

        Args:
            images: List of N PIL RGB images.

        Returns:
            (N, dim) float32 numpy array, L2-normalized.
        """
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embs = self.model.get_image_features(**inputs)
            embs = F.normalize(embs, dim=-1)
        return embs.cpu().float().numpy()

    def _encode_texts_batch(self, texts: List[str]) -> np.ndarray:
        """Batch encode text descriptions to normalized embeddings.

        Args:
            texts: List of N text strings.

        Returns:
            (N, dim) float32 numpy array, L2-normalized.
        """
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            embs = self.model.get_text_features(**inputs)
            embs = F.normalize(embs, dim=-1)
        return embs.cpu().float().numpy()

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

    def _encode_text(self, text: str) -> np.ndarray:
        """Encode a text description to a normalized embedding vector.

        Args:
            text: Visual description string (e.g. from VLM Stage 1).

        Returns:
            (dim,) float32 numpy array, L2-normalized.
        """
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            embedding = self.model.get_text_features(**inputs)
            embedding = F.normalize(embedding, dim=-1)

        return embedding.cpu().numpy().squeeze()
