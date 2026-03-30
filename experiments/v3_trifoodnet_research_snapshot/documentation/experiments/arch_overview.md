# Architecture Overview: TriFoodNet

TriFoodNet is a hierarchical vision-language system designed for automated food recognition. The system decomposes the task into three specialized stages:

## 1. Stage 1: Visual Grounding (Qwen2.5-VL-3B)
- **Objective**: Detect food items and provide initial category proposals.
- **Input**: Raw RGB Image.
- **Output**: Bounding boxes and coarse labels in JSON format.
- **Optimization**: Fine-tuned with **rsLoRA** to adapt the VLM's grounding capabilities to cafeteria environments.

## 2. Stage 2: Promptable Segmentation (SAM 3 Hiera-Large)
- **Objective**: Generate pixel-precise masks for each detected item.
- **Input**: Stage 1 Bounding Boxes + Original Image.
- **Output**: High-resolution instance masks.
- **Novelty**: Strictly box-prompted flow (no text prompts) to ensure geometric consistency across food textures.

## 3. Stage 3: In-Context Classification (PictSure ICL Transformer)
- **Objective**: Refine classification using a support set of exemplars.
- **Mechanism**: Few-shot In-Context Learning (ICL) using an attention-based Transformer head.
- **Optimization**: **LoRA-augmented** (r=32) for domain-specific menu adaptation.
