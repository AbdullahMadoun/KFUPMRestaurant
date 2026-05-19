# Experiment 01: FoodSAM + PictSure Hybrid Pipeline

**Branch:** `main` (legacy V1 content)  
**Timeline:** Early project phase  
**Status:** Superseded by V2 and V3

## Approach
The first attempt at food detection used a hybrid pipeline combining:
- **FoodSAM** (a food-specialized variant of SAM) for segmentation
- **PictSure** for food classification using DINOv2 embeddings
- **MMSeg** (OpenMMLab) for semantic segmentation backbone

## Key Files
- `src/pipeline.py` — Main inference pipeline
- `src/mmseg/` — Full MMSeg integration for semantic segmentation
- `requirements.txt` — Dependencies

## Outcome
This approach was abandoned because FoodSAM's generic food segmentation did not align with the cafeteria-tray domain (Saudi-specific dishes). The pipeline lacked the ability to handle the closed 32-class taxonomy needed for billing.

## Relevance to Final Paper
This experiment established the foundational idea of a multi-stage pipeline (detect → segment → classify) that persisted through all subsequent versions.
