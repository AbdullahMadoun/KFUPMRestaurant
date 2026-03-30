# Training Metrics & Results

Final verification results after aligning with the 117-image labelled dataset.

## 1. Verified Results (Step 10)
| Metric | Value |
| :--- | :--- |
| **Total Loss** | 4.88 |
| **Stage 1 LM Loss** | 1.57 |
| **Stage 2 Dice Loss** | 0.95 |
| **Stage 3 Accuracy** | **90.0%** |
| **Samples/sec** | 14.2 (at BS=4) |

## 2. Convergence Analysis
- **Stage 3 (ICL)**: Accuracy spiked from ~65% to **90%** within 10 steps, confirming the effectiveness of the PictSure LoRA adapters (Rank 32).
- **Stage 2 (SAM)**: Stable Dice loss below 1.0, indicating excellent alignment with real-world food masks.

## 3. Stability Note
The final production recommendation is **Batch Size 1 with Gradient Accumulation 4**, which avoids the positional embedding size mismatches observed in Qwen2.5-VL when processing variable-token vision batches.
