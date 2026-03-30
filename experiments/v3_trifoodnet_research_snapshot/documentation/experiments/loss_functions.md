# Loss Functions & Multi-Task Objective

TriFoodNet optimizes a composite objective function across three distinct vision tasks.

## 1. Total Loss Formula
$$ \mathcal{L}_{total} = \lambda_1 \mathcal{L}_{LM} + \lambda_2 ( \mathcal{L}_{BCE} + \mathcal{L}_{Dice} ) + \lambda_3 \mathcal{L}_{CE} $$

## 2. Weighted Balancing
- **$\lambda_1 = 1.0$**: Grounding loss (Qwen).
- **$\lambda_2 = 1.0$**: Mask precision loss (SAM 3).
- **$\lambda_3 = 1.5$**: Classification priority (PictSure). **Set to 1.5 to prioritize categorical accuracy for menu items.**

## 3. Stage-Specific Loss Details
- **Stage 1**: Standard cross-entropy for detection JSON generation (Language Modeling Loss).
- **Stage 2**:
  - **BCE**: Pixel-wise binary cross-entropy.
  - **Dice**: Intersection-over-Union (IoU) based loss for handling class imbalance (small items).
- **Stage 3**: CE (Cross-Entropy) for the 29-way classification head, augmented by the episodic support set similarity.
