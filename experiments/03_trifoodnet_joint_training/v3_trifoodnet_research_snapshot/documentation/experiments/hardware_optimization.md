# Hardware Optimization: NVIDIA RTX 5090 (Blackwell)

To leverage the 32GB memory and Blackwell architecture, several low-level optimizations were applied:

## 1. Precision & Compilation
- **BF16 (Bfloat16)**: Native support enabled across all 3 stages, providing a ~2.1x throughput increase over FP32 with minimal precision loss.
- **Torch Inductor (Compile)**: `torch.compile` is applied to the Stage 3 PictSure head, generating optimized Triton kernels for the episodic attention layers.

## 2. Memory Scaling
| Config | Batch Size | VRAM Usage | Status |
| :--- | :--- | :--- | :--- |
| **Stable** | 1 | 12.8 GB | Best for Stage 1 VLM stability. |
| **Optimal** | 4 | 26.4 GB | Peak throughput (14.2 s/s). |
| **Unstable** | 8 | >32 GB | OutOfMemory (OOM) during graph capture. |

## 3. Contiguity & Device Alignment
- **Contiguity**: Added `.contiguous()` before view operations in the PictSure embedding layer to resolve non-contiguous memory layouts common in high-throughput 5090 kernels.
- **Global Device Matching**: Ensure all episodic support sets and attention masks are strictly aligned on `cuda:0` via dynamic `.to(device)` logic.
