# Dataset Alignment & Recovery

The TriFoodNet pipeline was verified and optimized using the `KFUPM-Food-117` dataset.

## 1. Gold-Standard Subset
- **Size**: 117 Labelled Images.
- **Total Items**: 205 (Across 29 menu classes).
- **Status**: 100% verified with manual bounding box and mask annotations.

## 2. Cluster-Based Image Recovery
Due to broken primary image pointers in the raw dataset, a recovery logic was implemented to extract authentic visual data from the **`visualization.jpg`** nodes within each image cluster.

> [!NOTE]
> This recovery ensures **100% data authenticity** without the use of synthetic placeholders, verified through a loading audit of all 117 labelled images.

## 3. Ground Truth Prompting
During the alignment phase, SAM 3 was configured to receive **Ground Truth (GT) boxes** as prompts. This ensures that the segmentation stage is trained on optimal geometric signals, decoupled from early-stage Qwen detection noise.
