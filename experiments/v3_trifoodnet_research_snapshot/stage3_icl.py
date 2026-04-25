# =============================================================================
# FILE: stage3_icl.py
# CATEGORY: ARCH
# PURPOSE: Stage 3 few-shot food classification through the upstream PictSure model.
# DEPENDENCIES: item_processing.py
# USED BY: benchmark_runtime.py, check_trainable.py, pipeline.py, run_dev_inference.py, run_single_inference.py, test_pictsure_lora.py, tests/test_allocation.py, train_joint.py, validate_pipeline_contracts.py, visualize_val_predictions.py
# KEY CLASSES/FUNCTIONS: _import_pictsure, FoodClassifier
# LAST MODIFIED: 2026-03-21T14:33:23.198257+00:00
# SNAPSHOT NOTES: no major inline issues detected during snapshot generation
# =============================================================================
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from item_processing import pil_images_to_tensor


def _import_pictsure():
    try:
        from PictSure import PictSure  # type: ignore
        from PictSure.normalization import normalize_samples  # type: ignore
        return PictSure, normalize_samples
    except ImportError:
        local_repo = Path(__file__).resolve().parent / "pictsure_library"
        if str(local_repo) not in sys.path:
            sys.path.insert(0, str(local_repo))
        from PictSure import PictSure  # type: ignore
        from PictSure.normalization import normalize_samples  # type: ignore
        return PictSure, normalize_samples


PictSure, normalize_samples = _import_pictsure()


DEFAULT_CLASSES = [
    "rice",
    "chicken",
    "meat",
    "lemon",
    "pasta",
    "vegetables_steamed",
    "fish",
    "okra",
    "Meat_kebab",
    "Aseeda_brown",
    "Aseeda_white",
    "cake",
    "Chicken_kebab",
    "soup",
    "Fasoolia",
    "hummus",
    "Koshari",
    "Muttabal",
    "tabboulah",
    "Bashamil",
    "mussaqaa",
    "salad",
    "bread",
    "dessert",
    "fries",
    "juice",
    "soup'",
    "soup_lenitel",
    "Tomatoe_Veggies",
]


# --- Snapshot note: Core Stage 3 module: few-shot support/query classification built on PictSure. ---
class FoodClassifier(nn.Module):
    """
    Adapter around the pretrained PictSure model.

    Training uses masked teacher-forced item crops with ground-truth labels.
    Inference uses masked crops generated from SAM3 predictions.
    """

    def __init__(
        self,
        clip_model: str = "pictsure/pictsure-vit",
        num_layers: int = 4,  # preserved for compatibility
        num_heads: int = 8,   # preserved for compatibility
        ff_dim: int = 1024,   # preserved for compatibility
        dropout: float = 0.1, # preserved for compatibility
        lora_cfg: Optional[dict] = None,
        *,
        num_classes: Optional[int] = None,
        class_names: Optional[Sequence[str]] = None,
        train_embedding: bool = True,
        inference_n_way: int = 5,
        inference_k_shot: int = 5,
    ):
        super().__init__()

        resolved_class_names = list(class_names or DEFAULT_CLASSES)
        resolved_num_classes = max(int(num_classes or len(resolved_class_names)), len(resolved_class_names))

        print(f"Loading pretrained PictSure model: {clip_model}")
        self.pictsure_model = PictSure.from_pretrained(clip_model, device="cpu")
        self._expand_num_classes(resolved_num_classes)

        self._lora_enabled = bool(lora_cfg and lora_cfg.get("enabled", False))
        if self._lora_enabled:
            from peft import LoraConfig, get_peft_model

            peft_config = LoraConfig(
                r=lora_cfg.get("r", 16),
                lora_alpha=lora_cfg.get("alpha", 32),
                target_modules=lora_cfg.get("target_modules", ["linear1", "linear2", "out_proj"]),
                lora_dropout=lora_cfg.get("dropout", 0.05),
                bias="none",
            )
            wrapped_transformer = get_peft_model(self.pictsure_model.transformer, peft_config)
            self.pictsure_model.transformer = wrapped_transformer
            print(f"Applied PEFT LoRA to PictSure transformer (r={lora_cfg.get('r', 16)})")

        self.icl = self.pictsure_model.transformer
        self.class_names = resolved_class_names
        self.num_classes = resolved_num_classes
        self.train_embedding = bool(train_embedding)
        self.inference_n_way = max(1, int(inference_n_way))
        self.inference_k_shot = max(1, int(inference_k_shot))
        self._support_index_by_class: Dict[int, list[int]] = {}
        self._cached_support_embeddings: Optional[torch.Tensor] = None
        self._cached_support_prototypes: Optional[torch.Tensor] = None
        self._cached_support_class_ids: Optional[torch.Tensor] = None
        self._cached_support_device: Optional[str] = None
        # Versioning so the support cache invalidates when weights change.
        self._support_cache_version: int = 0
        self._set_trainability()

    def _expand_num_classes(self, target_num_classes: int):
        current_num_classes = int(getattr(self.pictsure_model, "num_classes", 0))
        if current_num_classes >= target_num_classes:
            self.pictsure_model.num_classes = current_num_classes
            return

        old_y = self.pictsure_model.y_projection
        new_y = nn.Linear(target_num_classes, old_y.out_features, bias=old_y.bias is not None)
        with torch.no_grad():
            new_y.weight.zero_()
            new_y.weight[:, :current_num_classes] = old_y.weight
            if old_y.bias is not None:
                new_y.bias.copy_(old_y.bias)

        old_fc = self.pictsure_model.fc
        new_fc = nn.Linear(old_fc.in_features, target_num_classes, bias=old_fc.bias is not None)
        with torch.no_grad():
            nn.init.xavier_uniform_(new_fc.weight)
            new_fc.weight[:current_num_classes] = old_fc.weight
            if old_fc.bias is not None:
                new_fc.bias.zero_()
                new_fc.bias[:current_num_classes] = old_fc.bias

        self.pictsure_model.y_projection = new_y.to(old_y.weight.device)
        self.pictsure_model.fc = new_fc.to(old_fc.weight.device)
        self.pictsure_model.num_classes = target_num_classes

    def _set_trainability(self):
        """Decide which Stage 3 params get gradients.

        With LoRA enabled, peft has already frozen the transformer's base weights
        and exposed only the LoRA adapters as trainable. We must NOT undo that —
        previously this method blindly set every param to require_grad=True,
        which trained the full base transformer in addition to the LoRA, defeating
        the LoRA-only-training intent.

        With LoRA disabled, full Stage 3 fine-tunes — the legacy behavior.

        ``train_embedding=False`` always freezes the encoder regardless of LoRA.
        """
        if not self._lora_enabled:
            # No LoRA → full transformer + (optionally) embedding train.
            for parameter in self.pictsure_model.parameters():
                parameter.requires_grad_(True)
        # else: keep peft's freeze-base-train-LoRA configuration.
        # Encoder freeze still respected in either mode.
        if not self.train_embedding:
            for parameter in self.pictsure_model.embedding.parameters():
                parameter.requires_grad_(False)

    @property
    def model_type(self) -> str:
        return str(getattr(self.pictsure_model, "embedding_model", "vit"))

    def _prepare_query_tensor(self, query_image) -> torch.Tensor:
        if isinstance(query_image, Image.Image):
            query_tensor = pil_images_to_tensor([query_image])[0]
        elif isinstance(query_image, torch.Tensor):
            query_tensor = query_image.detach()
            if query_tensor.ndim == 4:
                query_tensor = query_tensor.squeeze(0)
            if query_tensor.max().item() > 1.0:
                query_tensor = query_tensor.float() / 255.0
            else:
                query_tensor = query_tensor.float()
        else:
            raise TypeError(f"Unsupported query type: {type(query_image)!r}")
        return query_tensor

    def _normalize_episode(
        self,
        support_images: torch.Tensor,
        query_images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        support_shape = support_images.shape
        query_shape = query_images.shape
        support_norm = normalize_samples(
            support_images,
            model_type=self.model_type,
            resize=(224, 224),
        )
        query_norm = normalize_samples(
            query_images,
            model_type=self.model_type,
            resize=(224, 224),
        )
        if support_norm.ndim == 4:
            support_norm = support_norm.view(
                support_shape[0],
                support_shape[1],
                support_norm.shape[-3],
                support_norm.shape[-2],
                support_norm.shape[-1],
            )
        if query_norm.ndim == 4:
            query_norm = query_norm.view(
                query_shape[0],
                query_shape[1],
                query_norm.shape[-3],
                query_norm.shape[-2],
                query_norm.shape[-1],
            )
        if query_norm.ndim == 3:
            query_norm = query_norm.unsqueeze(0).unsqueeze(0)
        return support_norm, query_norm

    def _build_class_mask(self, support_labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        mask = torch.full_like(logits, -1e4)
        for row_index in range(support_labels.shape[0]):
            active_ids = torch.unique(support_labels[row_index]).long()
            active_ids = active_ids[(active_ids >= 0) & (active_ids < logits.shape[1])]
            if active_ids.numel() == 0:
                mask[row_index].zero_()
                continue
            mask[row_index, active_ids] = 0.0
        return mask

    def forward(
        self,
        support_images: torch.Tensor,
        query_images: torch.Tensor,
        support_labels: torch.Tensor,
        n_way: Optional[int] = None,
        k_shot: Optional[int] = None,
    ) -> torch.Tensor:
        support_images = support_images.float()
        query_images = query_images.float()
        support_labels = support_labels.long()

        if support_images.ndim != 5:
            raise ValueError(f"support_images must be [B, NK, C, H, W], got {tuple(support_images.shape)}")
        if query_images.ndim != 5:
            raise ValueError(f"query_images must be [B, NQ, C, H, W], got {tuple(query_images.shape)}")
        if support_labels.ndim == 1:
            support_labels = support_labels.unsqueeze(0).expand(support_images.shape[0], -1)

        device = next(self.pictsure_model.parameters()).device
        support_norm, query_norm = self._normalize_episode(support_images, query_images)
        support_norm = support_norm.to(device)
        query_norm = query_norm.to(device)
        support_labels = support_labels.to(device)

        batch_size, support_count, channels, height, width = support_norm.shape
        _, query_count, _, _, _ = query_norm.shape

        expanded_support = support_norm.unsqueeze(1).expand(
            batch_size,
            query_count,
            support_count,
            channels,
            height,
            width,
        ).reshape(batch_size * query_count, support_count, channels, height, width)
        expanded_labels = support_labels.unsqueeze(1).expand(
            batch_size,
            query_count,
            support_count,
        ).reshape(batch_size * query_count, support_count)
        flattened_queries = query_norm.reshape(batch_size * query_count, 1, channels, height, width)

        logits = self.pictsure_model(expanded_support, expanded_labels, flattened_queries, embedd=True)
        class_mask = self._build_class_mask(expanded_labels, logits)
        return logits + class_mask

    def _embed_samples(self, images: torch.Tensor, device: torch.device) -> torch.Tensor:
        if images.ndim == 4:
            images = images.unsqueeze(0)
        batch_size, num_images = images.shape[:2]
        normalized = normalize_samples(
            images.float(),
            model_type=self.model_type,
            resize=(224, 224),
        ).to(device)
        if normalized.ndim == 3:
            normalized = normalized.unsqueeze(0).unsqueeze(0)
        if normalized.ndim == 4:
            normalized = normalized.view(
                batch_size,
                num_images,
                normalized.shape[-3],
                normalized.shape[-2],
                normalized.shape[-1],
            )
        embedded = self.pictsure_model.embedding(normalized)
        return embedded.view(-1, embedded.shape[-1])

    def invalidate_support_cache(self):
        """Drop cached support embeddings/prototypes. Call this after any weight
        update to the embedding (e.g., end of epoch, post-checkpoint-load) so the
        cosine retriever re-embeds against the current encoder. Without this,
        retrieval silently uses stale embeddings from before training started."""
        self._cached_support_embeddings = None
        self._cached_support_prototypes = None
        self._cached_support_class_ids = None
        self._cached_support_device = None
        self._support_cache_version += 1

    def _ensure_support_cache(self, device: torch.device):
        if (
            self._cached_support_device == str(device)
            and self._cached_support_embeddings is not None
            and self._cached_support_prototypes is not None
            and self._cached_support_class_ids is not None
        ):
            return

        support_images = self.support_images.to(device)
        support_labels = self.support_labels.to(device)
        support_embeddings = self._embed_samples(support_images, device)

        class_ids = torch.unique(support_labels).long()
        prototypes = []
        for class_id in class_ids.tolist():
            class_mask = support_labels == class_id
            prototypes.append(support_embeddings[class_mask].mean(dim=0))

        self._cached_support_embeddings = support_embeddings
        self._cached_support_prototypes = torch.stack(prototypes, dim=0)
        self._cached_support_class_ids = class_ids
        self._cached_support_device = str(device)

    def _select_support_subset(
        self,
        query_tensor: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Pick the top-K most-similar classes to the query and return their support
        images + labels. Also returns the class ids the retriever selected so
        callers can audit retrieval recall vs transformer accuracy separately.

        Returns:
            support_images, support_labels, selected_class_ids (sorted by descending similarity)
        """
        support_images = self.support_images.to(device)
        support_labels = self.support_labels.to(device)
        unique_labels = torch.unique(support_labels)

        if unique_labels.numel() <= self.inference_n_way:
            # No retrieval narrowing needed — every class is a candidate.
            return support_images, support_labels, [int(c) for c in unique_labels.tolist()]

        self._ensure_support_cache(device)
        assert self._cached_support_prototypes is not None
        assert self._cached_support_class_ids is not None

        query_embedding = self._embed_samples(query_tensor.unsqueeze(0), device)[0]
        similarities = F.cosine_similarity(
            self._cached_support_prototypes,
            query_embedding.unsqueeze(0),
            dim=-1,
        )
        top_count = min(self.inference_n_way, similarities.numel())
        top_indices = torch.topk(similarities, k=top_count).indices
        selected_class_ids = self._cached_support_class_ids[top_indices].tolist()

        selected_indices: list[int] = []
        for class_id in selected_class_ids:
            class_indices = self._support_index_by_class.get(int(class_id), [])
            selected_indices.extend(class_indices[: self.inference_k_shot])

        if not selected_indices:
            return support_images, support_labels, [int(c) for c in unique_labels.tolist()]

        index_tensor = torch.tensor(selected_indices, dtype=torch.long, device=device)
        return (
            support_images.index_select(0, index_tensor),
            support_labels.index_select(0, index_tensor),
            [int(c) for c in selected_class_ids],
        )

    def set_support_set(
        self,
        support_images: Sequence[Image.Image] | torch.Tensor,
        support_labels: Sequence[int] | torch.Tensor,
    ):
        if isinstance(support_images, torch.Tensor):
            tensor_images = support_images.float()
            if tensor_images.ndim == 5:
                tensor_images = tensor_images.squeeze(0)
        else:
            tensor_images = pil_images_to_tensor(list(support_images))

        if isinstance(support_labels, torch.Tensor):
            tensor_labels = support_labels.long()
            if tensor_labels.ndim > 1:
                tensor_labels = tensor_labels.squeeze(0)
        else:
            tensor_labels = torch.tensor(list(support_labels), dtype=torch.long)

        if tensor_images.ndim != 4:
            raise ValueError(f"support_images must resolve to [N, C, H, W], got {tuple(tensor_images.shape)}")
        if tensor_labels.ndim != 1:
            raise ValueError(f"support_labels must resolve to [N], got {tuple(tensor_labels.shape)}")
        if tensor_images.shape[0] != tensor_labels.shape[0]:
            raise ValueError("support_images and support_labels must contain the same number of items.")

        self._support_index_by_class = {}
        for index, class_id in enumerate(tensor_labels.tolist()):
            self._support_index_by_class.setdefault(int(class_id), []).append(index)
        self._cached_support_embeddings = None
        self._cached_support_prototypes = None
        self._cached_support_class_ids = None
        self._cached_support_device = None

        if "support_images" in self._buffers:
            self._buffers["support_images"] = tensor_images
        else:
            self.register_buffer("support_images", tensor_images, persistent=False)
        if "support_labels" in self._buffers:
            self._buffers["support_labels"] = tensor_labels
        else:
            self.register_buffer("support_labels", tensor_labels, persistent=False)

    def compile_transformer(self, **compile_kwargs):
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this runtime.")
        compiled = torch.compile(self.pictsure_model.transformer, **compile_kwargs)
        self.pictsure_model.transformer = compiled
        self.icl = compiled
        return compiled

    @torch.inference_mode()
    def classify(
        self,
        query_image,
        device: str = "cuda",
        top_k: int = 1,
    ):
        if not hasattr(self, "support_images") or not hasattr(self, "support_labels"):
            raise RuntimeError("Support set is not loaded. Call set_support_set() before classify().")

        runtime_device = torch.device(device)
        if runtime_device.type == "cuda" and not torch.cuda.is_available():
            runtime_device = torch.device("cpu")
        self.to(runtime_device)

        query_tensor = self._prepare_query_tensor(query_image).to(runtime_device)
        support_images, support_labels, selected_class_ids = self._select_support_subset(
            query_tensor, runtime_device
        )

        # Diagnostic: which classes did the cosine retriever surface? Caller
        # (pipeline.run) reads this attribute right after classify() returns,
        # then attaches it to the FoodItem so eval can compute retrieval recall.
        self.last_candidate_class_ids: list[int] = list(selected_class_ids)
        self.last_candidate_class_names: list[str] = [
            self.class_names[cid] if 0 <= cid < len(self.class_names) else f"class_{cid}"
            for cid in selected_class_ids
        ]

        # Keep inference on the trainable transformer ICL path so the
        # deployed classifier matches the model optimized during training.
        self.eval()
        logits = self.forward(
            support_images=support_images.unsqueeze(0),
            query_images=query_tensor.unsqueeze(0).unsqueeze(0),
            support_labels=support_labels.unsqueeze(0),
        )[0]
        probabilities = torch.softmax(logits, dim=-1)
        values, indices = probabilities.topk(k=max(1, top_k))

        predictions = []
        for score, class_index in zip(values.tolist(), indices.tolist()):
            class_name = (
                self.class_names[class_index]
                if 0 <= class_index < len(self.class_names)
                else f"class_{class_index}"
            )
            predictions.append((class_name, float(score)))
        return predictions
