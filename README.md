# TahmajaNet-VLM-for-Detecting-Classifying-and-Billing-Food

## Project Metadata
### Authors
- **Team:** Anonymous Authors / KFUPM Restaurant Project
- **Supervisor Name:** Dr. Muzammil Behzad
- **Affiliations:** KFUPM

## Introduction
Generating an itemized bill from a single photo of a cafeteria tray is a deceptively hard vision problem. The output is not a set of pixel-accurate masks, nor a ranked list of detection scores: it is a *multiset of food classes* priced by a downstream lookup. A single missed item or a single duplicated detection is a wrong bill. 

TahamajaNet tackles this by breaking the process into a three-stage distillation pipeline that connects the broad understanding of Large Vision-Language Models (VLMs) with the speed of smaller, specialized models. We:
1. Run a 72B-parameter teacher model offline to automatically label a massive dataset of cafeteria trays.
2. Distill this capability into a fast 7B-parameter student model that runs at inference time to generate precise bounding boxes.
3. Use a frozen segmenter (SAM3) and a DINOv2 vector-database classifier to confidently identify the food and generate the bill.

The goal: an end-to-end detection system that perfectly counts and classifies food items, enabling automated checkout lines without sacrificing latency.

## Problem Statement
We treat the task as producing an exact *multiset of food classes* (count-exact and class-matched) from a single tray image. We face two main problems: off-the-shelf generative VLMs are too slow and often "overcount" items, while traditional open-set detectors lack the deep understanding required to distinguish subtle differences (e.g., merging toppings with the base dish while ignoring plates).

Our questions:
Q1: Can a 72B VLM act as an automated labeling engine to bootstrap a smaller 7B model?
Q2: How do we fix the "cardinality regression" (overcounting) that happens when fine-tuning generative VLMs for detection?
Q3: How do we evaluate this accurately when standard Intersection over Union (IoU) metrics fail to capture the billing reality?

We will report our results using a custom end-to-end metric called **Dish-Correct**, evaluating the count-exact match and class multiset match, comparing a baseline 7B model against our Distilled 7B model.

## Application Area and Project Domain
Targets include automated cafeteria checkout lines, smart restaurants, and autonomous billing systems. Users need: a clear, itemized bill of exactly what is on their tray, processed in under 2 seconds.

Our pipeline bridges the gap between massive, slow conversational VLMs and fast, rigid traditional detectors, making it perfectly suited for high-throughput, closed-taxonomy environments like university or corporate cafeterias.

## What is the paper trying to do, and what are you planning to do?
We propose **TahamajaNet**, a three-stage detection pipeline. During the offline phase, a massive Qwen2.5-VL-72B model acts as a data engine to label 4,999 tray images, refined by SAM3 and clustered by DINOv2+SigLIP. 

During the online inference phase (what runs in production), we use a distilled Qwen2.5-VL-7B student model fine-tuned with a novel *cardinality-aware structural loss* to arrest overcounting. This student emits bounding boxes, which are passed to a frozen SAM3 for mask generation. Finally, the masked crops are classified using a DINOv2-large $k$-NN reference bank.

We plan to demonstrate that this pipeline increases the end-to-end "Dish-Correct" rate from 39.4% (baseline) to **86.4%**, completely eliminating zero-detection failures and solving the cardinality bottleneck.

### Project Documents
- **Paper Latex Files:** [Paper](/paper)

### Reference Paper
- [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/abs/2408.12966)
- [Segment Anything Model 3 (SAM3)](https://arxiv.org/abs/2402.15391)

### Reference GitHub
- [Qwen-VL Official Repository](https://github.com/QwenLM/Qwen-VL)

### Reference Dataset
- **v3.2 Dataset:** A 4,629-item, 32-class hand-curated Saudi cafeteria-tray dataset.

## Project Technicalities

### Terminologies
- **Vision-Language Model (VLM):** A model capable of understanding both image and text inputs to generate text outputs (like bounding box coordinates).
- **Knowledge Distillation:** Training a smaller "student" model to replicate the behavior of a much larger "teacher" model.
- **Cardinality-Aware Loss:** A custom loss function that penalizes the model when the predicted *number* of items differs from the ground truth.
- **Dish-Correct Metric:** Our end-to-end evaluation metric requiring both exact count and exact class multiset matches.
- **k-NN Reference Bank:** A vector database of embeddings where new images are classified by finding the "nearest neighbor" in the pre-embedded training set.

### Problem Statements
- **Problem 1:** 72B VLMs are too slow ($>5$ seconds per image) and expensive for real-time cafeteria checkout.
- **Problem 2:** 7B VLMs hallucinate bounding boxes, leading to severe overcounting and incorrect bills.
- **Problem 3:** Standard metrics like mAP or IoU do not correlate with a correct monetary bill.

### Loopholes or Research Areas
- **Evaluation Metrics:** Bounding box IoU does not matter for billing; identifying the exact item does.
- **Data Scarcity:** No existing dataset covers the specific 32-class taxonomy of a Saudi cafeteria (e.g., kabsa rice, asidah, specific wrapped items).

### Problem vs. Ideation: Proposed 3 Ideas to Solve the Problems
1. **Offline 72B Data Engine:** Use the massive 72B model just once offline to label a high-quality dataset, bypassing the need for manual bounding-box annotation.
2. **Cardinality Structural Loss:** Add a specific loss term during the 7B student fine-tuning that penalizes absolute count error ($|\hat{n} - n|$).
3. **Retrieval-Based Classification:** Use a frozen DINOv2 embedding space with a $k$-NN search instead of a standard linear classifier head to better handle visual variations (like different types of rice).

### Proposed Solution: Code-Based Implementation
This repository provides the complete historical and finalized codebase for TahamajaNet.
- **Stage 1:** Distilled Qwen2.5-VL-7B running optimized inference.
- **Stage 2:** SAM3 integration for mask refinement.
- **Stage 3:** DINOv2-large vector database classifier.

### Key Components

**Final Pipeline (`src/tahamajanet/`):**
- `pipeline.py`: End-to-end 3-stage inference orchestrator
- `stage1_qwen.py`: VLM bounding box generation (inference)
- `stage1_kcfd/`: Stage 1 training module — model, dataset, **cardinality-aware loss**, trainer (14 files)
- `stage2_sam.py`: SAM3 mask refinement with confidence cascade
- `stage3_vector_db.py`: DINOv2 k-NN maximum-similarity retrieval classifier
- `losses.py` / `metrics.py`: Loss functions and Dish-Correct evaluation metrics
- `train_joint.py` / `train.py`: Training entry points
- `master_config.yaml`: Complete training configuration

**Experiment History (`experiments/`):**

| # | Folder | Approach | Paper Section |
|---|--------|----------|---------------|
| 01 | `foodsam_pictsure_hybrid` | FoodSAM + PictSure legacy pipeline | — |
| 02 | `sam3_qwen_detection` | SAM3 + Qwen-VL first integration | §3 ancestor |
| 03 | `trifoodnet_joint_training` | Joint multi-task training (3 stages) | §5 observations |
| 04 | `three_stage_batch_prototype` | Batch inference with 7B/3B model testing | §3.1, §7 data |
| 05 | `72b_teacher_evaluation` | 72B AWQ teacher on 809 images | §3 (Data Engine) |
| 06 | `vlm_grounded_prompting` | Text+bbox vs bbox-only prompt testing | §4.1 design |
| 07 | `stage1_distillation_training` | **Cardinality-aware 7B fine-tuning** | §4, §5, §7 ★ |
| 08 | `training_infrastructure` | Handoff docs, dataset spec, deployment | §3.4, §4.5 |
| 09 | `dinov2_knn_classifier` | DINOv2 max-sim retrieval (final Stage 3) | §4.3 ★ |
| 10 | `config_driven_refactor` | Config-driven V2 pipeline architecture | — |

See [`docs/PAPER_MAPPING.md`](docs/PAPER_MAPPING.md) for the complete section-by-section mapping.

**Paper (`paper/`):**
- `paper.tex` + `sections/` — Full LaTeX source for the TahamajaNet paper

## Model Workflow
1. **Input:** A cafeteria tray image is passed to the system.
2. **Stage 1 (Grounding):** The distilled Qwen2.5-VL-7B model processes the image and emits a strict JSON format containing bounding boxes for every distinct food item.
3. **Stage 2 (Segmentation):** SAM3 takes the image and bounding boxes, returning pixel-accurate masks, suppressing background noise.
4. **Stage 3 (Classification):** Masked crops are embedded using DINOv2-large. The embedding is compared against a reference bank of 3,909 training crops using Cosine Similarity. The label of the closest match is assigned.
5. **Output:** A finalized multiset of identified dishes ready for billing.

## How to Run the Code

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/BRAIN-Lab-AI/TahmajaNet-VLM-for-Detecting-Classifying-and-Billing-Food.git
    cd TahmajaNet-VLM-for-Detecting-Classifying-and-Billing-Food
    ```

2. **Set Up the Environment:**
    Create a virtual environment and install dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r src/tahamajanet/requirements.txt
    ```

3. **Run Inference:**
    ```bash
    python src/tahamajanet/pipeline.py --image_path path/to/tray.jpg
    ```

## Acknowledgments
- **Open-Source Communities:** Thanks to QwenLM, Meta (SAM/DINOv2), and Hugging Face.
- **Resource Providers:** Compute via vast.ai (H100 SXM 80GB).
