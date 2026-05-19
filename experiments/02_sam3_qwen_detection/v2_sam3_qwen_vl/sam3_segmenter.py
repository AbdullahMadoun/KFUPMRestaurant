import os
import torch
import numpy as np
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

class SAM3Segmenter:
    """
    Wrapper for the SAM3 (Segment Anything Model 3) to handle food item segmentation.
    
    Features:
    - Text-prompted segmentation using discrete prompts.
    - Dynamic Thresholding: Automatically lowers confidence requirement if no masks are found.
    - Global NMS: Suppresses overlapping detections across different prompts to keep the top 5 objects.
    """
    def __init__(self, model_path="facebook/sam3", device="cuda", confidence_threshold=0.1):
        self.device = device
        self.confidence_threshold = confidence_threshold
        import sam3
        # Try to locate the bpe file in known locations
        possible_paths = [
            # Known location from find command
            "/root/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
            # Relative to current dir if sam3 repo is here
            os.path.abspath(os.path.join("sam3", "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")),
            # Relative to package if installed
            os.path.join(os.path.dirname(sam3.__file__), "assets", "bpe_simple_vocab_16e6.txt.gz")
        ]
        
        bpe_path = None
        for path in possible_paths:
            if os.path.exists(path):
                bpe_path = path
                break
        
        if bpe_path is None:
             raise FileNotFoundError("Could not find bpe_simple_vocab_16e6.txt.gz in expected locations.")

        print(f"Loading SAM3 model with bpe_path: {bpe_path}")
        self.model = build_sam3_image_model(bpe_path=bpe_path, device=device)
        self.processor = Sam3Processor(self.model, device=device, confidence_threshold=confidence_threshold)

    def segment(self, image_path, prompts, max_objects=5, iou_threshold=0.7):
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return []

        image = Image.open(image_path).convert("RGB")
        
        # Initial threshold
        current_threshold = self.confidence_threshold
        
        # Possible fallback thresholds if 0 masks are found
        fallbacks = [0.05, 0.02, 0.01]
        
        all_detections = []
        
        # Loop for dynamic thresholding
        thresholds_to_try = [current_threshold] + [f for f in fallbacks if f < current_threshold]
        
        for thresh in thresholds_to_try:
            print(f"Trying threshold: {thresh}")
            # Update processor threshold
            self.processor.confidence_threshold = thresh
            
            all_detections = []
            for prompt in prompts:
                print(f"Segmenting prompt: {prompt}")
                state = self.processor.set_image(image)
                state = self.processor.set_text_prompt(prompt, state)
                
                if "masks" in state and state["masks"] is not None:
                    masks = state["masks"]
                    boxes = state["boxes"] if "boxes" in state else torch.zeros((masks.shape[0], 4))
                    scores = state["scores"] if "scores" in state else torch.zeros(masks.shape[0])
                    
                    for i in range(masks.shape[0]):
                        all_detections.append({
                            "label": prompt,
                            "mask": masks[i].squeeze(),
                            "box": boxes[i],
                            "score": float(scores[i])
                        })
            
            if all_detections:
                print(f"Found {len(all_detections)} candidates at threshold {thresh}.")
                break
            else:
                print(f"No candidates found at threshold {thresh}.")

        if not all_detections:
            return []

        # Sort by score descending
        all_detections = sorted(all_detections, key=lambda x: x["score"], reverse=True)
        
        keep = []
        
        def compute_iou(mask1, mask2):
            intersection = torch.logical_and(mask1, mask2).sum().float()
            union = torch.logical_or(mask1, mask2).sum().float()
            if union == 0: return 0
            return intersection / union

        for det in all_detections:
            if len(keep) >= max_objects:
                break
                
            discard = False
            for kept_det in keep:
                iou = compute_iou(det["mask"], kept_det["mask"])
                if iou > iou_threshold:
                    discard = True
                    break
            
            if not discard:
                keep.append(det)

        # Re-grouping for visualizer
        grouped_results = {}
        for det in keep:
            label = det["label"]
            if label not in grouped_results:
                grouped_results[label] = {"label": label, "masks": [], "boxes": [], "scores": []}
            grouped_results[label]["masks"].append(det["mask"].unsqueeze(0))
            grouped_results[label]["boxes"].append(det["box"].unsqueeze(0))
            grouped_results[label]["scores"].append(det["score"])

        final_results = []
        for label, res in grouped_results.items():
            final_results.append({
                "label": label,
                "masks": torch.cat(res["masks"]).cpu().numpy(),
                "boxes": torch.cat(res["boxes"]).cpu().numpy(),
                "scores": np.array(res["scores"])
            })
            
        return final_results
