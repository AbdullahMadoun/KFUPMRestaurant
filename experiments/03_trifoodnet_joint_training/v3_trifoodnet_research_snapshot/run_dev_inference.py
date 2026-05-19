# =============================================================================
# FILE: run_dev_inference.py
# CATEGORY: INFER
# PURPOSE: Development-set inference and visualization script.
# DEPENDENCIES: config_loader.py, dataset_integration.py, pipeline.py, stage1_qwen.py, stage2_sam.py, stage3_icl.py
# USED BY: visualize_val_predictions.py
# KEY CLASSES/FUNCTIONS: resolve_device, build_bnb_config, run_inference
# LAST MODIFIED: 2026-03-21T14:29:48.059593+00:00
# SNAPSHOT NOTES: uses cfg.run.name for checkpoint lookup; split='val' is preserved as a legacy alias for dev
# =============================================================================

import os
import sys
import torch
import json
from pathlib import Path
from PIL import Image
import numpy as np

# Add current dir to path
sys.path.append(os.getcwd())

from pipeline import TriFoodNet
from stage1_qwen import QwenGrounder
from stage2_sam import SAM3Segmenter
from stage3_icl import FoodClassifier
from config_loader import load_config
from dataset_integration import (
    JointFoodDataset,
    build_class_name_index,
    build_export_paths,
    load_masked_item_image,
    read_json,
)
from torch.utils.data import DataLoader

def resolve_device(requested: str | None):
    name = str(requested or "auto").strip().lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def build_bnb_config(cfg, device):
    if not getattr(cfg.hardware, "load_in_4bit", False):
        return None
    if device.type != "cuda":
        return None
    from transformers import BitsAndBytesConfig
    compute_dtype = torch.bfloat16 if getattr(cfg.hardware, "bf16", False) else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def run_inference():
    # 1. Setup
    cfg = load_config("master_config.yaml")
    device = resolve_device(getattr(cfg.hardware, "device", "auto"))
    import stage3_icl
    print(f"Stage 3 module path: {stage3_icl.__file__}")
    print(f"Using device: {device}")

    c1 = cfg.stage1
    c2 = cfg.stage2
    c3 = cfg.stage3
    h = cfg.hardware

    bnb_config = build_bnb_config(cfg, device)
    export_paths = build_export_paths(
        cfg.data.integration.batch_root,
        export_root=(cfg.data.integration.export_root or None),
        repo_root=(cfg.data.integration.repo_root or None),
    )
    class_records = read_json(export_paths.export_root / "classes.json")
    class_names, class_name_to_id = build_class_name_index(
        export_paths.export_root,
        classes=class_records if isinstance(class_records, list) else None,
    )

    # 2. Initialize Stages
    print("Initializing Stage 1 (Qwen) in 4-bit...")
    stage1 = QwenGrounder(
        model_name=c1.model_name,
        lora_r=c1.lora.r,
        lora_alpha=c1.lora.alpha,
        lora_dropout=c1.lora.dropout,
        lora_target_modules=c1.lora.target_modules,
        use_rslora=c1.lora.get("use_rslora", False),
        gradient_checkpointing=h.gradient_checkpointing,
        quantization_config=bnb_config,
    )
    
    print("Initializing Stage 2 (SAM 3) in 4-bit...")
    stage2 = SAM3Segmenter(
        model_name=c2.model_name,
        freeze_image_encoder=c2.freeze.image_encoder,
        freeze_prompt_encoder=c2.freeze.prompt_encoder,
        gradient_checkpointing=h.gradient_checkpointing,
        quantization_config=bnb_config,
    )
    
    print("Initializing Stage 3 (PictSure)...")
    stage3 = FoodClassifier(
        clip_model=c3.clip_model,
        num_layers=c3.transformer.num_layers,
        num_heads=c3.transformer.num_heads,
        ff_dim=c3.transformer.ff_dim,
        dropout=c3.transformer.dropout,
        lora_cfg=getattr(c3, "lora", None),
        num_classes=max(int(cfg.data.num_classes), len(class_names), (max(class_name_to_id.values()) + 1) if class_name_to_id else 0),
        class_names=class_names,
        train_embedding=bool(getattr(c3, "train_embedding", True)),
        inference_n_way=int(c3.eval.n_way),
        inference_k_shot=int(c3.eval.k_shot),
    ).to(device)

    # 3. Assemble Pipeline
    pipeline = TriFoodNet(stage1, stage2, stage3)
    
    # 3b. Load Best Checkpoint if available
    ckpt_path = Path(cfg.paths.checkpoints) / cfg.run.name / "joint" / "best"
    if ckpt_path.exists():
        print(f"Loading trained weights from {ckpt_path}...")
        pipeline.load(str(ckpt_path))
    else:
        print("No 'best' checkpoint found. Running with foundation weights.")

    # (Note: Stage 1 and 2 are already on GPU via bnb device_map="auto")
    pipeline.eval()
    
    # 4. Load Dataset (Validation Split)
    dataset = JointFoodDataset(
        batch_root=cfg.data.integration.batch_root,
        export_root=cfg.data.integration.export_root,
        split="val",
        image_size=cfg.data.image_size,
        split_seed=cfg.data.integration.split_seed,
    )
    print(f"Validation Dataset Loaded: {len(dataset)} images.")

    # 5. Build Support Set (1-shot) from Training Split
    print("Building 1-shot support set from training split...")
    train_dataset = JointFoodDataset(
        batch_root=cfg.data.integration.batch_root,
        export_root=cfg.data.integration.export_root,
        split="train",
        image_size=cfg.data.image_size,
        split_seed=cfg.data.integration.split_seed,
    )
    
    # Simple 1-shot: pick first occurrence of each class in train_dataset
    class_to_img = {}
    support_images_list = []
    support_labels_list = []
    
    for row in train_dataset.stage3_rows:
        label_id = row["class_id"] # categorical index from classes.json
        if label_id not in class_to_img and len(class_to_img) < int(cfg.data.num_classes):
            img = load_masked_item_image(row, train_dataset.paths.batch_root, train_dataset.paths.repo_root)
            if img:
                support_images_list.append(img)
                support_labels_list.append(label_id)
                class_to_img[label_id] = True
                
    if support_images_list:
        pipeline.stage3.set_support_set(support_images_list, support_labels_list)
    
    # 6. Results Directory
    results_dir = Path("results/dev")
    results_dir.mkdir(parents=True, exist_ok=True)

    # 6. Inference Loop
    all_results = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            batch = dataset[i] # Get direct item
            image_id = batch["image_id"]
            title = batch.get("notes", "No notes")
            print(f"[{i+1}/{len(dataset)}] Processing {image_id}...")
            
            # Predict (passing PIL image)
            output = pipeline.run(
                pil_image=batch["pil_image"],
                image_id=image_id
            )
            
            # 6. Save Visualization (NEW)
            try:
                from PIL import ImageDraw, ImageFont
                viz_img = batch["pil_image"].copy().convert("RGBA")
                draw = ImageDraw.Draw(viz_img)
                
                for item in output.items:
                    box = item.box  # [x1, y1, x2, y2]
                    label = f"{item.label} ({item.confidence:.2f})"
                    
                    # Draw Box
                    draw.rectangle(box, outline="red", width=3)
                    
                    # Draw Mask if it exists
                    if item.mask is not None:
                        mask_np = (item.mask.detach().cpu().numpy().astype(np.uint8) * 128)
                        mask_img = Image.fromarray(mask_np, mode='L')
                        
                        # Create a color layer (e.g., green for mask)
                        color_layer = Image.new("RGBA", viz_img.size, (0, 255, 0, 0))
                        viz_img = Image.composite(Image.new("RGBA", viz_img.size, (0, 255, 0, 100)), viz_img, mask_img)
                        draw = ImageDraw.Draw(viz_img) # Reset draw after composite
                    
                    # Draw Label
                    draw.text((box[0], box[1] - 10), label, fill="yellow")

                viz_path = results_dir / f"{image_id}_viz.png"
                viz_img.convert("RGB").save(viz_path)
                print(f"[{i+1}/{len(dataset)}] Saved visualization → {viz_path.name}")
            except Exception as e:
                print(f"[{i+1}/{len(dataset)}] Visualization failed: {e}")
            
            # Save JSON entry
            result_item = {
                "image_id": image_id,
                "notes": title,
                "total_price": output.total_price,
                "items": [
                    {
                        "label": item.label,
                        "confidence": float(item.confidence),
                        "box": item.box,
                        "price": item.price
                    } for item in output.items
                ],
                "latency_ms": output.latency_ms
            }
            all_results.append(result_item)

            # Save Aggregate JSON (Incremental)
            with open(results_dir / "inference_results.json", "w") as f:
                json.dump(all_results, f, indent=2)
            
            print(f"[{i+1}/{len(dataset)}] Completed {image_id}.")

    print(f"Inference Complete. Final results saved to {results_dir}")

if __name__ == "__main__":
    run_inference()
