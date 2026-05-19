import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def get_random_color():
    # Generate saturated, bold colors by ensuring at least one channel is high and one is low
    channels = [random.randint(0, 50), random.randint(180, 255), random.randint(0, 255)]
    random.shuffle(channels)
    return tuple(channels)

def apply_mask(image, mask, color, alpha=0.7):
    """Apply the given mask to the image."""
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image

def visualize(image_path, results, output_path="result.png", draw_boxes=True, alpha=0.7, thickness=3):
    """
    Renders segmentation masks and labels onto the original image.
    
    Args:
        image_path (str): Path to the source image.
        results (list): List of detection dictionaries (label, masks, scores).
        output_path (str): Path to save the annotated image.
        draw_boxes (bool): Whether to draw bounding boxes.
        alpha (float): Opacity of the segmentation masks (0-1).
        thickness (int): Thickness of the mask boundaries (contours).
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image {image_path}")
        return

    output = image.copy()
    overlay = image.copy()
    
    # First pass: Draw all masks and boundaries on overlay
    for item in results:
        masks = item['masks']
        if masks.shape[0] > 0:
            color = get_random_color()
            item['color'] = color
            
            for i in range(masks.shape[0]):
                mask = masks[i].squeeze()
                if mask.ndim > 2: mask = mask[0] # Safety
                mask_bool = mask > 0
                
                # Draw mask on overlay
                for c in range(3):
                    overlay[:, :, c] = np.where(mask_bool,
                                              color[c],
                                              overlay[:, :, c])
                
                # Draw thick boundary (contour) on the overlay too, so it gets blended or kept sharp
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, color, thickness)

    # Blend overlay with original image
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    # Second pass: Draw boxes (optional) and labels on the blended image (so they are opaque)
    for item in results:
        masks = item['masks']
        boxes = item['boxes']
        label = item['label']
        color = item.get('color', (0, 255, 0)) # Fallback if no mask was drawn
        
        for i in range(masks.shape[0]):
            box = boxes[i] if len(boxes) > i else None
             
            if box is not None and draw_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
                
                # Include count in label if multiple segments for this prompt
                num_masks = masks.shape[0]
                label_text = f"{label}"
                if num_masks > 1:
                    label_text += f" {i+1}/{num_masks}"
                    
                cv2.putText(output, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            else:
                 # If no box, find contour center from mask
                mask = masks[i]
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if contours: 
                     # approximate center
                     M = cv2.moments(contours[0])
                     if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        
                        num_masks = masks.shape[0]
                        label_text = f"{label}"
                        if num_masks > 1:
                            label_text += f" {i+1}/{num_masks}"
                            
                        cv2.putText(output, label_text, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imwrite(output_path, output)
    print(f"Saved visualization to {output_path}")
