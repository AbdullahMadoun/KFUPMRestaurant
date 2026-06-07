# 🍔 TahmajaNet: VLM for Detecting, Classifying, and Billing Food

![Version](https://img.shields.io/badge/Version-1.0-blue)
![Model](https://img.shields.io/badge/Model-Qwen2.5_VL-orange)
![Pipeline](https://img.shields.io/badge/Pipeline-3_Stage-success)

## 📌 Project Metadata
- **Authors:** Anonymous Authors / KFUPM Restaurant Project
- **Supervisor:** Dr. Muzammil Behzad
- **Affiliation:** KFUPM

---

## 📖 Introduction

Generating an itemized bill from a single photo of a cafeteria tray is a deceptively hard vision problem. The output is not a set of pixel-accurate masks, nor a ranked list of detection scores: it is a *multiset of food classes* priced by a downstream lookup. A single missed item or a duplicated detection results in an incorrect bill. 

**TahmajaNet** tackles this by breaking the process into a three-stage distillation pipeline:
1. **Teacher Labeling:** A 72B-parameter teacher model operates offline to auto-label a massive dataset of cafeteria trays.
2. **Student Distillation:** This capability is distilled into a fast 7B-parameter student model that runs at inference time to generate precise bounding boxes.
3. **Segmentation & Classification:** A frozen segmenter (SAM3) and a DINOv2 vector-database classifier confidently identify the food and generate the bill.

**The Goal:** An end-to-end detection system that perfectly counts and classifies food items, enabling automated checkout lines without sacrificing latency.

---

## 🎯 Problem Statement

Off-the-shelf generative VLMs are too slow and often "overcount" items, while traditional open-set detectors lack the deep understanding required to distinguish subtle differences (e.g., merging toppings with the base dish while ignoring plates).

**Key Questions Addressed:**
1. Can a 72B VLM act as an automated labeling engine to bootstrap a smaller 7B model?
2. How do we fix the "cardinality regression" (overcounting) in generative VLMs?
3. How do we evaluate accurately when standard IoU metrics fail to capture the billing reality?

We evaluate using a custom end-to-end Exact Subset Accuracy metric (which we term **Dish-Correct**), achieving an increase from 39.4% (baseline) to **86.4%**.

---

## 🏢 Application Area

Targets include **automated cafeteria checkout lines**, **smart restaurants**, and **autonomous billing systems**.
Our pipeline bridges the gap between massive, slow conversational VLMs and fast, rigid traditional detectors, making it perfectly suited for high-throughput, closed-taxonomy environments.

---

## 🛠️ The Pipeline

TahamajaNet is a three-stage detection pipeline:

1. **Stage 1 (Grounding):** A distilled Qwen2.5-VL-7B student model fine-tuned with a novel *cardinality-aware structural loss* emits bounding boxes.
2. **Stage 2 (Segmentation):** SAM3 refines the bounding boxes into pixel-accurate masks.
3. **Stage 3 (Classification):** Masked crops are classified using a DINOv2-large $k$-NN reference bank via Cosine Similarity.

---

## 📂 Key Resources

- 📄 **Paper LaTeX Files:** [Paper](/paper)
- 📚 **Reference Papers:**
  - [Qwen2-VL: Enhancing Vision-Language Model's Perception](https://arxiv.org/abs/2408.12966)
  - [Segment Anything Model 3 (SAM3)](https://arxiv.org/abs/2402.15391)
- 💾 **Reference Dataset:** A 4,938-item, 32-class hand-curated Saudi cafeteria-tray dataset (v3.2).

---

## 🧬 Technical Implementation

### Key Innovations
- **Offline 72B Data Engine:** Bypassing manual annotation using an offline Qwen2.5-VL-72B model.
- **Cardinality Structural Loss:** A custom loss penalizing absolute count error ($|\hat{n} - n|$).
- **Retrieval-Based Classification:** Using a frozen DINOv2 embedding space with a $k$-NN search.

### Repository Structure
**Final Pipeline (`src/tahamajanet/`):**
- `pipeline.py`: End-to-end 3-stage inference orchestrator.
- `stage1_qwen.py`: VLM bounding box generation (inference).
- `stage1_kcfd/`: Stage 1 training module.
- `stage2_sam.py`: SAM3 mask refinement.
- `stage3_vector_db.py`: DINOv2 k-NN retrieval classifier.
- `losses.py` / `metrics.py`: Exact Subset Accuracy (Dish-Correct) evaluation metrics.

**Paper (`paper/`):**
- `paper.tex` & `sections/` — Full LaTeX source for the TahamajaNet paper.

See [`docs/PAPER_MAPPING.md`](docs/PAPER_MAPPING.md) for the complete section-by-section mapping of historical experiments.

---

## 🚀 How to Run

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/BRAIN-Lab-AI/TahmajaNet-VLM-for-Detecting-Classifying-and-Billing-Food.git
    cd TahmajaNet-VLM-for-Detecting-Classifying-and-Billing-Food
    ```

2. **Set Up the Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r src/tahamajanet/requirements.txt
    ```

3. **Run Inference:**
    ```bash
    python src/tahamajanet/pipeline.py --image_path path/to/tray.jpg
    ```

---

## 🙏 Acknowledgments
- **Open-Source Communities:** QwenLM, Meta (SAM/DINOv2), and Hugging Face.
- **Compute:** vast.ai (H100 SXM 80GB).
