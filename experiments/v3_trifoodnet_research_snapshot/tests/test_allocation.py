# =============================================================================
# FILE: tests/test_allocation.py
# CATEGORY: TEST
# PURPOSE: Snapshot-retained source file for test_allocation.py.
# DEPENDENCIES: losses.py, stage3_icl.py
# USED BY: None
# KEY CLASSES/FUNCTIONS: test_stage3_allocation_and_backward
# LAST MODIFIED: 2026-03-21T07:23:37.035724+00:00
# SNAPSHOT NOTES: no major inline issues detected during snapshot generation
# =============================================================================
import torch
from stage3_icl import FoodClassifier
from losses import Stage3Loss

def test_stage3_allocation_and_backward():
    """
    Simulates a dry-run of the Stage 3 training dynamics to ensure torch best practices
    (like no graph leaks and appropriate gradient detachment).
    """
    device = torch.device('cpu')
    
    # Tiny dummy models to mock training allocation logic
    model = FoodClassifier(
        clip_model="openai/clip-vit-base-patch32", 
        num_layers=1, 
        num_heads=2, 
        ff_dim=128
    ).to(device)
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=True)
    loss_fn = Stage3Loss(kind="cross_entropy")

    batch_size = 2
    n_way = 3
    k_shot = 2
    q_per_class = 1
    support_labels = torch.arange(n_way).repeat_interleave(k_shot).unsqueeze(0).repeat(batch_size, 1)
    
    # Fabricate memory
    support = torch.randn(batch_size, n_way * k_shot, 3, 224, 224)
    query = torch.randn(batch_size, n_way * q_per_class, 3, 224, 224)
    labels = torch.randint(0, n_way, (batch_size * n_way * q_per_class,))
    
    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    # Optimizer step 1
    optimizer.zero_grad(set_to_none=True)
    logits = model(support, query, support_labels=support_labels, n_way=n_way, k_shot=k_shot)
    loss, loss_dict = loss_fn(logits, labels)
    
    assert loss.requires_grad, "Loss tensor detached unintentionally. Backprop will fail."
    loss.backward()
    
    # Check that grads exist and are sensible (best practice)
    has_grads = any(p.grad is not None for p in model.parameters())
    assert has_grads, "No gradients were computed!"
    
    optimizer.step()
    
    # Optimizer step 2 (ensure no retained graph exception)
    optimizer.zero_grad(set_to_none=True)
    logits2 = model(support, query, support_labels=support_labels, n_way=n_way, k_shot=k_shot)
    loss2, loss2_dict = loss_fn(logits2, labels)
    loss2.backward()
    
    if torch.cuda.is_available():
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()
        print(f"CUDA memory managed gracefully. End allocation: {final_memory / 1e6:.2f} MB")
        
    print("Allocation & memory flow tests passed.")

if __name__ == "__main__":
    test_stage3_allocation_and_backward()
