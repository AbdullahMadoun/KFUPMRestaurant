# =============================================================================
# FILE: train_stage3_hf.py
# CATEGORY: TRAIN
# PURPOSE: Alternate Stage 3 training entry point preserved for research experiments.
# DEPENDENCIES: config_loader.py, dataset_integration.py, losses.py
# USED BY: None
# KEY CLASSES/FUNCTIONS: OfficialPictSureHFTrainerModel, DummyProcessor, DummyClipProcessor, main
# LAST MODIFIED: 2026-03-21T07:47:50.152219+00:00
# SNAPSHOT NOTES: appears stale against the current JointBatchCollator signature
# =============================================================================
import os
import sys
import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments

# We now import the authentic model from the official pip package instead of our Stage 3 replica!
from PictSure import PictSure

from losses import Stage3Loss
from dataset_integration import JointFoodDataset, JointBatchCollator
from config_loader import load_config

class OfficialPictSureHFTrainerModel(nn.Module):
    """
    Wraps the REAL PictSure library architecture to fit flawlessly into the Hugging Face Trainer.
    Only the ICL Transformer layers are updated.
    """
    def __init__(self, loss_fn: Stage3Loss, n_way: int, k_shot: int):
        super().__init__()
        self.pictsure_model = PictSure.from_pretrained("pictsure/pictsure-vit")
        
        # Explicitly guarantee the vision backbone remains completely frozen
        for p in self.pictsure_model.embedding.parameters():
            p.requires_grad_(False)
            
        self.loss_fn = loss_fn
        self.n_way = n_way
        self.k_shot = k_shot

    def forward(
        self, 
        support_images=None, 
        query_images=None, 
        query_labels=None, 
        return_dict=True, 
        **kwargs
    ):
        device = support_images.device
        B, NK, C, H, W = support_images.shape
        _, NQ, _, _, _ = query_images.shape
        
        # The authentic library passes queries strictly as singletons appended to the sequence.
        # Flat repeat to fit the native [B * NQ] API context window sizes
        x_train = support_images.unsqueeze(1).expand(B, NQ, NK, C, H, W).reshape(B * NQ, NK, C, H, W)
        x_pred = query_images.reshape(B * NQ, 1, C, H, W)
        
        y_train_single = torch.repeat_interleave(torch.arange(self.n_way, device=device), self.k_shot)
        y_train = y_train_single.unsqueeze(0).expand(B * NQ, -1)
        
        logits = self.pictsure_model(x_train, y_train, x_pred, embedd=True)
        episode_logits = logits[:, :self.n_way]
        
        loss = None
        if query_labels is not None:
            flat_labels = query_labels.reshape(-1)
            sample_per_class = torch.tensor([self.k_shot] * self.n_way, device=device).float()
            loss, metrics = self.loss_fn(episode_logits, flat_labels, sample_per_class=sample_per_class)
            
        if not return_dict:
            return (loss, episode_logits) if loss is not None else episode_logits
            
        return {"loss": loss, "logits": episode_logits}

class DummyProcessor:
    def __call__(self, *args, **kwargs):
        return {"pixel_values": torch.zeros(1), "image_grid_thw": torch.zeros(1)}
        
class DummyClipProcessor:
    def __init__(self):
        from transformers import CLIPProcessor
        self.proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    def __call__(self, *args, **kwargs):
        return self.proc(*args, **kwargs)

def main():
    overrides = sys.argv[1:] or None
    cfg = load_config(None, overrides)
    c3 = cfg.stage3
    integration = cfg.data.integration
    
    if not getattr(integration, "batch_root", ""):
        raise ValueError("Set `data.integration.batch_root` before running this.")
    
    print(f"Initializing REAL PictSure model utilizing pictsure/pictsure-vit")
    
    loss_fn = Stage3Loss(
        label_smoothing=c3.training.label_smoothing,
        kind=getattr(c3.loss, "name", "cross_entropy"),
        logit_adjust_tau=getattr(c3.loss, "logit_adjust_tau", 1.0),
    )
    
    wrapped_model = OfficialPictSureHFTrainerModel(
        loss_fn=loss_fn,
        n_way=c3.episode.n_way, 
        k_shot=c3.episode.k_shot
    )
    
    train_ds = JointFoodDataset(
        batch_root=integration.batch_root,
        export_root=(integration.export_root or None),
        repo_root=(integration.repo_root or None),
        split="train",
        image_size=cfg.data.image_size,
        train_ratio=integration.train_ratio,
        val_ratio=integration.val_ratio,
        test_ratio=integration.test_ratio,
        split_seed=integration.split_seed,
        n_way=c3.episode.n_way,
        k_shot=c3.episode.k_shot,
        query_per_class=c3.episode.query_per_class,
    )
    
    val_ds = JointFoodDataset(
        batch_root=integration.batch_root,
        export_root=(integration.export_root or None),
        repo_root=(integration.repo_root or None),
        split="val",
        image_size=cfg.data.image_size,
        train_ratio=integration.train_ratio,
        val_ratio=integration.val_ratio,
        test_ratio=integration.test_ratio,
        split_seed=integration.split_seed,
        n_way=c3.eval.n_way,
        k_shot=c3.eval.k_shot,
        query_per_class=1,
    )
    
    collator = JointBatchCollator(
        stage1_processor=DummyProcessor(),
        stage3_processor=DummyClipProcessor(),
        stage1_prompt="",
    )
    
    def hf_collate_fn(batch):
        b = collator(batch)
        return {
            "support_images": b["support_images"],
            "query_images": b["query_images"],
            "query_labels": b["query_labels"]
        }
        
    training_args = TrainingArguments(
        output_dir="outputs/stage3_pictsure_hf",
        num_train_epochs=c3.training.epochs,
        per_device_train_batch_size=cfg.joint.training.batch_size,
        per_device_eval_batch_size=cfg.joint.training.batch_size,
        learning_rate=c3.training.learning_rate,
        weight_decay=c3.training.weight_decay,
        logging_dir="logs/stage3_pictsure_hf",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        remove_unused_columns=False,
        report_to="none"
    )
    
    trainer = Trainer(
        model=wrapped_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=hf_collate_fn,
    )
    
    print("Starting exact HF finetuning of Official PictSure ICL transformer...")
    trainer.train()
    
    os.makedirs("outputs/stage3_pictsure_hf", exist_ok=True)
    torch.save(wrapped_model.pictsure_model.transformer.state_dict(), "outputs/stage3_pictsure_hf/best_icl.pt")
    print("Training complete. ICL weights saved.")

if __name__ == "__main__":
    main()
