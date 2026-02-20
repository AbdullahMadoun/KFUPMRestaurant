"""FAISS vector store: build from reference images, save/load, query."""

import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger("pipeline")


class VectorStore:
    """FAISS flat inner-product index over SigLIP 2 image embeddings."""

    def __init__(self):
        self.index = None
        self.metadata: List[dict] = []

    def build(self, menu_dir: str, schema: dict, model, processor, device: str):
        """Build index from reference images organized in per-item subdirectories.

        Args:
            menu_dir: Directory with subdirectories per menu item, each containing .jpg images.
            schema: Menu schema dict, e.g. {"chicken": {"category": "protein", "price": 15.0, ...}}.
            model: SigLIP 2 model (already loaded).
            processor: SigLIP 2 processor.
            device: "cuda" or "cpu".
        """
        import faiss

        embeddings = []
        metadata = []
        menu_path = Path(menu_dir)

        for item_name, item_info in schema.items():
            item_dir = menu_path / item_name
            if not item_dir.exists():
                logger.warning(f"Reference directory not found for '{item_name}': {item_dir}")
                continue

            image_paths = list(item_dir.glob("*.jpg")) + list(item_dir.glob("*.png")) + list(item_dir.glob("*.jpeg"))
            if not image_paths:
                logger.warning(f"No images found in {item_dir}")
                continue

            for img_path in image_paths:
                emb = self._encode_image(img_path, model, processor, device)
                if emb is not None:
                    embeddings.append(emb)
                    metadata.append({
                        "name": item_name,
                        "category": item_info.get("category", "unknown"),
                        "price": item_info.get("price", 0.0),
                        "source_image": str(img_path),
                    })
                    logger.info(f"Encoded: {img_path.name} -> {item_name}")

        if not embeddings:
            raise ValueError("No embeddings were generated. Check menu_dir and schema.")

        # Stack and normalize
        emb_array = np.stack(embeddings).astype("float32")
        faiss.normalize_L2(emb_array)

        # Build flat inner-product index (cosine similarity on normalized vectors)
        dim = emb_array.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(emb_array)
        self.metadata = metadata

        logger.info(f"Built FAISS index: {len(metadata)} vectors, dim={dim}")

    def save(self, index_path: str, metadata_path: str):
        """Save index and metadata to disk."""
        import faiss

        if self.index is None:
            raise RuntimeError("No index to save. Call build() first.")

        faiss.write_index(self.index, index_path)
        Path(metadata_path).write_text(json.dumps(self.metadata, indent=2))
        logger.info(f"Saved index to {index_path}, metadata to {metadata_path}")

    def load(self, index_path: str, metadata_path: str):
        """Load index and metadata from disk."""
        import faiss

        if not Path(index_path).exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not Path(metadata_path).exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        self.index = faiss.read_index(index_path)
        self.metadata = json.loads(Path(metadata_path).read_text())
        logger.info(f"Loaded FAISS index: {self.index.ntotal} vectors, {len(self.metadata)} metadata entries")

    def query(self, embedding: np.ndarray, top_k: int = 3) -> List[Tuple[dict, float]]:
        """Query the index with a single embedding.

        Args:
            embedding: (1, dim) or (dim,) float32 array, L2-normalized.
            top_k: Number of nearest neighbors to return.

        Returns:
            List of (metadata_dict, similarity_score) tuples, sorted by descending score.
        """
        if self.index is None:
            raise RuntimeError("No index loaded. Call load() or build() first.")

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        embedding = embedding.astype("float32")

        # Clamp top_k to available vectors
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(embedding, k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0:
                continue
            results.append((self.metadata[idx], float(dist)))

        return results

    @staticmethod
    def _encode_image(image_path, model, processor, device: str) -> np.ndarray:
        """Encode a single image to a normalized embedding vector."""
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)

            with torch.no_grad():
                embedding = model.get_image_features(**inputs)
                embedding = F.normalize(embedding, dim=-1)

            return embedding.cpu().numpy().squeeze()
        except Exception as e:
            logger.warning(f"Failed to encode {image_path}: {e}")
            return None
