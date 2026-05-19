# Experiment 09: DINOv2 k-NN Retrieval Classifier

**Branch:** `vector-db-classifier`  
**Timeline:** Late research phase  
**Status:** **Final Stage 3 approach used in the paper**

## Approach
Replaced the earlier ICL (in-context learning) transformer classifier with a simpler, more effective approach:
- Embed every train-split masked crop with **frozen DINOv2-large** (1024-d, L2-normalized)
- Build a **reference bank** R ∈ ℝ^{3909 × 1024} from all training items
- At inference: embed the query crop, compute `argmax cosine_similarity(query, R)`, assign the label of the nearest reference

Key insight: max-similarity retrieval (k-NN top-1) outperforms mean-pooled prototypes because multimodal classes (e.g., "rice" contains kabsa, biryani, mandi) have distinct visual signatures that get averaged out by prototyping.

## Key Files
- `stage3_vector_db.py` — DINOv2 k-NN classifier implementation
- `VECTOR_DB.md` — Technical documentation for the vector database approach

## Outcome
The DINOv2 k-NN classifier achieved higher dish-correct than the ICL transformer, with the bonus of zero training (it's purely retrieval-based). The ablation in the paper shows max-sim retrieval gives +1.6pp over mean-pooled prototypes.

## Relevance to Final Paper
This is the exact Stage 3 classifier described in §4.3 (DINOv2 Maximum-Similarity Retrieval). The paper states: "Unlike the more common mean-pooled prototype, the k-NN top-1 formulation handles the multimodal classes."
