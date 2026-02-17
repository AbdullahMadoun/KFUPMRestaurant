import argparse
import os
import sys
import cv2
import numpy as np
import torch
import colorsys
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# Ensure local modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PictSure import PictSure

LABEL_MAP = {0: "Chicken", 1: "Fish", 2: "Rice"}

def get_distinct_color(idx, n=12):
    """Generate vibrant distinct colors using HLS."""
    h = (idx % n) / n
    l, s = 0.5, 0.9
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (int(r * 255), int(g * 255), int(b * 255))

def crop_for_classifier(image_rgb, mask):
    """Crop segment and composite onto neutral background for PictSure."""
    ys, xs = np.where(mask)
    if len(xs) == 0: return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    crop = image_rgb[y0:y1+1, x0:x1+1].copy()
    m = mask[y0:y1+1, x0:x1+1]
    bg = np.full_like(crop, 235) # Light gray bg
    bg[m] = crop[m]
    return Image.fromarray(bg)

def get_pictsure_model(device, assets_dir):
    print("\n📦 Loading PictSure model...")
    # Using public model without token if possible, or user needs to provide one via env var
    hf_token = os.environ.get("HF_TOKEN") 
    
    try:
        model = PictSure.from_pretrained("pictsure/pictsure-vit", token=hf_token).to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure you have set HF_TOKEN environment variable if accessing private models.")
        raise

    # Dynamic context paths
    context_dir = assets_dir / "context"
    context_config = [
        (context_dir / "CHICKEN.jpg", 0), 
        (context_dir / "FISH.jpg", 1), 
        (context_dir / "RICE.jpg", 2)
    ]
    
    imgs, labs = [], []
    for path, lab in context_config:
        if path.exists():
            imgs.append(Image.open(path).convert("RGB"))
            labs.append(lab)
        else:
            print(f"⚠️ Warning: Context image not found: {path}")
            
    if not imgs:
        raise RuntimeError(f"No context images found in {context_dir}!")
        
    model.set_context_images(imgs, labs)
    return model

def main():
    p = argparse.ArgumentParser(description="Food Analysis Pipeline: Hybrid FoodSAM + PictSure")
    p.add_argument("--input_dir", required=True, help="Directory containing FoodSAM output folders")
    p.add_argument("--output_dir", default="results", help="Directory to save final visualized results")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on")
    p.add_argument("--assets_dir", default="../assets", help="Path to assets directory (relative to script or absolute)")
    args = p.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent
    assets_dir = (script_dir / args.assets_dir).resolve()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    print(f"🚀 Starting Food Analysis Pipeline")
    print(f"   Input: {input_dir}")
    print(f"   Output: {output_dir}")
    print(f"   Assets: {assets_dir}")
    print(f"   Device: {args.device}")

    if not input_dir.exists():
        print(f"❌ Error: Input directory does not exist: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Model
    pm = get_pictsure_model(args.device, assets_dir)

    # Sort folders to ensure consistent order
    folders = sorted([f for f in input_dir.iterdir() if f.is_dir()])
    
    for folder in tqdm(folders, desc="Classifying images"):
        img_path = folder / "input.jpg"
        mask_path = folder / "enhance_mask.png"
        
        if not img_path.exists() or not mask_path.exists():
            continue

        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        enhance_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Find unique labels excluding 0 (background)
        unique_labels = np.unique(enhance_mask)
        segments = []
        for lab in unique_labels:
            if lab == 0: continue
            # Extract mask for this specific FoodSeg103 label
            lab_mask = (enhance_mask == lab).astype(np.uint8)
            # Find connected components (multiple items of same label)
            num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(lab_mask, connectivity=8)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < 450: continue # Skip noise
                seg_mask = (labels_im == i)
                segments.append(seg_mask)

        # Classify each segment using PictSure
        final_segments = []
        for i, seg in enumerate(segments):
            pil = crop_for_classifier(image, seg)
            if pil:
                with torch.no_grad():
                    pred, probs = pm.predict(pil, return_probs=True)
                final_segments.append({
                    "mask": seg,
                    "label": LABEL_MAP[pred],
                    "confidence": float(probs[pred].item()),
                    "color": get_distinct_color(len(final_segments))
                })

        # Final Visualization with Neon Colors & Labels
        vis = image.copy()
        for s in final_segments:
            mask = s["mask"]
            color = s["color"]
            
            # Draw semi-transparent overlay
            overlay = np.zeros_like(vis)
            overlay[mask] = color
            vis = cv2.addWeighted(vis, 1.0, overlay, 0.4, 0)
            
            # Draw thick contour
            cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, color, 3)
            
            # Label at center
            ys, xs = np.where(mask)
            if len(xs):
                cx, cy = int(xs.mean()), int(ys.mean())
                text = f"{s['label']} ({s['confidence']:.2f})"
                # Shadow
                cv2.putText(vis, text, (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 4)
                # Text
                cv2.putText(vis, text, (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        save_name = folder.name.replace(" ", "_")
        save_path = output_dir / f"{save_name}_hybrid_vis.jpg"
        cv2.imwrite(str(save_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        # print(f"  ✅ Saved hybrid result to {save_path}")

    print(f"\n✨ Processing Complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
