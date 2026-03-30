# =============================================================================
# FILE: run_single_inference.py
# CATEGORY: INFER
# PURPOSE: Single-image inference CLI around the full pipeline.
# DEPENDENCIES: config_loader.py, dataset_integration.py, pipeline.py, stage1_qwen.py, stage2_sam.py, stage3_icl.py
# USED BY: None
# KEY CLASSES/FUNCTIONS: run_single
# LAST MODIFIED: 2026-03-21T14:29:58.767925+00:00
# SNAPSHOT NOTES: uses cfg.run.name for checkpoint lookup; split='val' is preserved as a legacy alias for dev
# =============================================================================

import os
import sys
import torch
import json
import argparse
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
from dataset_integration import JointFoodDataset, build_export_paths, build_class_name_index

def run_single():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_id", type=str, required=True)
    args = parser.parse_args()

    # 1. Setup
    cfg = load_config("master_config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    export_paths = build_export_paths(
        cfg.data.integration.batch_root,
        export_root=(cfg.data.integration.export_root or None),
        repo_root=(cfg.data.integration.repo_root or None),
    )
    class_names, class_name_to_id = build_class_name_index(export_paths.export_root)

    bnb_config = None
    if getattr(cfg.hardware, "load_in_4bit", False) and device.type == "cuda":
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

    # 2. Initialize Stages
    stage1 = QwenGrounder(
        model_name=cfg.stage1.model_name,
        lora_r=cfg.stage1.lora.r,
        lora_alpha=cfg.stage1.lora.alpha,
        lora_target_modules=cfg.stage1.lora.target_modules,
        quantization_config=bnb_config,
    )
    stage2 = SAM3Segmenter(
        model_name=cfg.stage2.model_name,
        quantization_config=bnb_config,
    )
    stage3 = FoodClassifier(
        clip_model=cfg.stage3.clip_model,
        num_layers=cfg.stage3.transformer.num_layers,
        num_heads=cfg.stage3.transformer.num_heads,
        ff_dim=cfg.stage3.transformer.ff_dim,
        num_classes=max(int(cfg.data.num_classes), len(class_names), (max(class_name_to_id.values()) + 1) if class_name_to_id else 0),
        class_names=class_names,
        inference_n_way=int(cfg.stage3.eval.n_way),
        inference_k_shot=int(cfg.stage3.eval.k_shot),
    ).to(device)

    pipeline = TriFoodNet(stage1, stage2, stage3)
    ckpt_path = Path(cfg.paths.checkpoints) / cfg.run.name / "joint" / "best"
    if ckpt_path.exists():
        pipeline.load(str(ckpt_path))

    pipeline.eval()
    
    # 4. Load Dataset to find the image
    dataset = JointFoodDataset(
        batch_root=cfg.data.integration.batch_root,
        export_root=cfg.data.integration.export_root,
        split="val",
        image_size=cfg.data.image_size,
        split_seed=cfg.data.integration.split_seed,
    )
    
    batch = None
    for i in range(len(dataset)):
        if dataset[i]["image_id"] == args.image_id:
            batch = dataset[i]
            break
            
    if batch is None:
        print(f"Image {args.image_id} not found.")
        return

    # 5. Build Support Set (needed for Stage 3)
    train_dataset = JointFoodDataset(
        batch_root=cfg.data.integration.batch_root,
        export_root=cfg.data.integration.export_root,
        split="train",
        image_size=cfg.data.image_size,
        split_seed=cfg.data.integration.split_seed,
    )
    
    support_images_list, support_labels_list = [], []
    class_to_img = {}
    for row in train_dataset.stage3_rows:
        label_id = row["class_id"]
        if label_id not in class_to_img and len(class_to_img) < int(cfg.data.num_classes):
            from dataset_integration import load_image_with_pointer_support
            img = load_image_with_pointer_support(row["crop_path"], train_dataset.paths.batch_root, train_dataset.paths.repo_root)
            if img:
                inputs = pipeline.stage3.pictsure_model.processor(images=img, return_tensors="pt")
                support_images_list.append(inputs["pixel_values"][0])
                support_labels_list.append(torch.tensor(label_id))
                class_to_img[label_id] = True
                
    if support_images_list:
        pipeline.stage3.set_support_set(
            torch.stack(support_images_list).to(device),
            torch.stack(support_labels_list).to(device)
        )

    # 6. Run
    output = pipeline.run(pil_image=batch["pil_image"], image_id=args.image_id)
    
    # 7. Visualization
    results_dir = Path("results/dev")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    from PIL import ImageDraw
    viz_img = batch["pil_image"].copy().convert("RGBA")
    draw = ImageDraw.Draw(viz_img)
    for item in output.items:
        box = item.box
        label = f"{item.label} ({item.confidence:.2f})"
        draw.rectangle(box, outline="red", width=3)
        if item.mask is not None:
            mask_np = (item.mask.astype(np.uint8) * 128)
            mask_img = Image.fromarray(mask_np, mode='L')
            viz_img = Image.composite(Image.new("RGBA", viz_img.size, (0, 255, 0, 100)), viz_img, mask_img)
            draw = ImageDraw.Draw(viz_img)
        draw.text((box[0], box[1] - 10), label, fill="yellow")

    viz_path = results_dir / f"{args.image_id}_viz.png"
    viz_img.convert("RGB").save(viz_path)
    
    # Update JSON (append mode)
    json_path = results_dir / "inference_results.json"
    all_results = []
    if json_path.exists():
        with open(json_path, "r") as f:
            all_results = json.load(f)
            
    # Remove existing entry for this image if it exists
    all_results = [r for r in all_results if r["image_id"] != args.image_id]
    
    all_results.append({
        "image_id": args.image_id,
        "items": [{"label": it.label, "confidence": float(it.confidence), "box": it.box} for it in output.items],
        "latency_ms": output.latency_ms
    })
    
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    run_single()
