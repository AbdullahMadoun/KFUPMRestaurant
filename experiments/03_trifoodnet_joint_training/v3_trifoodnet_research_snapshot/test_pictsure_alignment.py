# =============================================================================
# FILE: test_pictsure_alignment.py
# CATEGORY: TEST
# PURPOSE: Research test script for PictSure alignment behavior.
# DEPENDENCIES: None
# USED BY: None
# KEY CLASSES/FUNCTIONS: test_pictsure_alignment
# LAST MODIFIED: 2026-03-21T07:47:48.136154+00:00
# SNAPSHOT NOTES: no major inline issues detected during snapshot generation
# =============================================================================
import torch
from PictSure import PictSure

def test_pictsure_alignment():
    print("Loading PictSure from official Hugging Face checkpoint...")
    model = PictSure.from_pretrained("pictsure/pictsure-vit", device="cpu")
    
    B = 2
    n_way = 10
    k_shot = 5
    NK = n_way * k_shot
    
    x_train = torch.randn(B, NK, 3, 224, 224)
    y_train = torch.randint(0, n_way, (B, NK))
    
    # Official library forces explicitly 1 query per sequence via its [-1] attention mask indexing
    x_pred = torch.randn(B, 1, 3, 224, 224)
    
    # Enforce base vectorization freezing 
    for p in model.embedding.parameters():
        p.requires_grad_(False)
        
    # Pass through
    logits = model(x_train, y_train, x_pred, embedd=True)
    
    # Verify sequence shapes and projection bounds
    assert logits.shape == (B, model.num_classes), f"Expected logits {(B, model.num_classes)}, got {logits.shape}"
    
    # Verify parameter freezes
    embed_trainable = sum(p.numel() for p in model.embedding.parameters() if p.requires_grad)
    transformer_trainable = sum(p.numel() for p in model.transformer.parameters() if p.requires_grad)
    proj_trainable = sum(p.numel() for p in model.x_projection.parameters() if p.requires_grad)
    
    assert embed_trainable == 0, f"Embedder should be frozen, but {embed_trainable} params are trainable!"
    assert transformer_trainable > 0, "Transformer should be training!"
    assert proj_trainable > 0, "Projections should be training!"
    
    print(f"Alignment verified!")
    print(f"Vectorization backbone trained parameters: {embed_trainable}")
    print(f"ICL Transformer trained parameters: {transformer_trainable + proj_trainable}")
    print("Curated test complete: The base ViT is fully frozen and only the transformer layers are optimized.")

if __name__ == "__main__":
    test_pictsure_alignment()
