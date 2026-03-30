# TriFoodNet: Cold Start & Usage Guide

This guide is designed to get a researcher or engineer quickly up to speed with the `TriFoodNet` pipeline, from cloning the repository to kicking off your first joint training run on a GPU instance.

---

## 1. Environment Setup & Authentication

The TriFoodNet pipeline relies on heavy Vision-Language Models (Qwen2.5-VL-3B) and advanced Segmentation models (SAM3). 

**Install Core Dependencies:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate datasets peft
pip install wandb pillow numpy pytest
```

**Hugging Face Authentication:**
Because the models (specifically Qwen and SAM3) may be gated or require token access, you must authenticate:
```bash
huggingface-cli login
```
*(Alternatively, you can provide your token via the `HF_TOKEN` environment variable so scripts authenticate seamlessly).*

---

## 2. Dataset Integration

TriFoodNet expects data in a specific multi-stage export format (`.jsonl` manifests). 

1. **Locate your dataset root**: Ensure your reviewed dataset `batch_results` folder is accessible. It should contain an `_review/dataset/` directory with `images_manifest.jsonl`, `stage1_item_detection.jsonl`, etc.
2. **Validate & Fix Dataset**: Run the automated dataset rules enforcement script to ensure that ONLY labeled images are used, and that your `qwen_bbox` aligns perfectly with your SAM tighter bounds:
   ```bash
   python fix_dataset.py
   ```
3. **Verify Constraints**:
   ```bash
   python -m pytest tests/test_dataset.py
   ```
   *If this passes, your data is 100% compliant with the TriFoodNet pipeline expectations.*

---

## 3. Configuration (`master_config.yaml`)

All logic is routed through `master_config.yaml`. 

1. **Set the Dataset Path**: Ensure `data.integration.batch_root` points to your dataset directory.
2. **Review the Hyperparameters**: The `master_config.yaml` is pre-tuned with baseline defaults:
   - **Stage 1 (Qwen)**: LoRA `r=16`, `alpha=32`, `lr=1.0e-4`.
   - **Stage 2 (SAM3)**: Decoder-only training, `lr=1.0e-4`.
   - **Stage 3 (ICL)**: `label_smoothing=0.1`, 4 layers.
   - **Joint**: `lr=1.0e-5` with `lambda=(1.0, 1.0, 1.0)`.
3. **Hardware Considerations**: 
   - A standard run requires **16GB to 24GB of VRAM**.
   - If you encounter Out-Of-Memory (OOM) errors, lower `stage1.training.batch_size` and `joint.training.batch_size` from `4` to `2`.

---

## 4. Pre-Flight Checks

Before commencing a long training loop, ensure the architecture correctly initializes on your hardware:

**1. Mock Allocation flow (CPU safe)**
Ensures PyTorch builds the graphs without memory leaks:
```bash
python -m tests.test_allocation
```

**2. SAM3 Checkpoint Download (CPU/GPU)**
Ensures your HuggingFace token resolves to the SAM3 repository smoothly:
```bash
python -m tests.test_sam3_allocation
```

**3. GPU Throughput Benchmark (GPU required)**
Estimates peak memory usage, throughput (samples/sec), and ensures no OOMs exist before actual checkpoints are saved:
```bash
python benchmark_runtime.py joint-train --steps 20 --warmup 5
```

---

## 5. Kicking Off Joint Training

Once your pre-flight checks pass, clear any cached VRAM and start the E2E training protocol:

```bash
python train_joint.py
```

**What happens during this script?**
1. Data is ingested dynamically via `dataset_integration.py`.
2. Stage 1 (Qwen) trains via LoRA using bounding box + LM text generation losses.
3. Stage 2 (SAM3) prompts the mask decoder using the generated bounding boxes and computes BCE+Dice loss.
4. Stage 3 (Classifier) computes Few-Shot Cross Entropy loss against pooled crops.
5. Gradients are backpropagated scaling by the `lambda` weights.
6. The `best` combined model is saved periodically in `checkpoints/v3-runX/joint/best/`.

---

## 6. Evaluation & Reporting

To visualize loss curves, evaluate model checkpoint degradations, or compute dataset summaries, generate the experiment report:

```bash
python experiment_report.py --run-dir logs/v3-run1/joint
```
*Outputs are serialized automatically into Markdown summaries, SVGs, and CSVs.*
