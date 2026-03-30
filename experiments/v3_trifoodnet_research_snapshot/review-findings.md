# Review Findings

Date: 2026-03-21

## Scope reviewed

- `/root/rest_model`
- `/root/pictsure_library`
- The exported dataset manifests under `/root/dataset/_review/dataset`

## Refactor status

The main pipeline was refactored to match the requested contract:

- Qwen remains the Stage 1 grounding model.
- SAM3 now remains the Stage 2 mask generator from those boxes.
- Stage 3 now consumes masked item crops instead of plain box crops.
- Training keeps teacher forcing by using GT boxes/masks/labels in the training data path.
- Inference keeps GT out of the loop and classifies SAM-masked crops.

## Refactored areas

### 1. Stage 3 now uses masked items instead of plain crops

Updated files:
- `/root/rest_model/item_processing.py`
- `/root/rest_model/dataset_integration.py`
- `/root/rest_model/pipeline.py`
- `/root/rest_model/run_dev_inference.py`

Summary:
- Added reusable masked-crop helpers.
- Changed the Stage 3 dataset path to build masked item images from `image_path + mask_path + bbox`.
- Changed inference to classify SAM-masked crops instead of raw box crops.

### 2. The PictSure wrapper was reworked around the pretrained model

Updated file:
- `/root/rest_model/stage3_icl.py`

Summary:
- Added import fallback to the local `/root/pictsure_library` checkout.
- Expanded the classifier head beyond the upstream 10-class config so dataset class IDs can be used directly.
- Wired LoRA onto the live transformer module instead of a detached alias.
- Removed the forced CPU fallback in `classify()`.
- Added support-set loading based on real class IDs and masked item tensors.

### 3. Teacher forcing and box flow are now explicit in the pipeline/trainer

Updated files:
- `/root/rest_model/pipeline.py`
- `/root/rest_model/train_joint.py`

Summary:
- `pipeline.forward()` now switches between GT boxes and Qwen-generated boxes instead of relying on a missing `pred_boxes` field.
- Stage 3 training now receives `support_labels` directly from the episodic dataset path.
- Runtime device resolution is now explicit and quantization is opt-in instead of being assumed.

### 4. SAM3 loading and trainability were cleaned up

Updated files:
- `/root/rest_model/stage2_sam.py`
- `/root/rest_model/master_config.yaml`

Summary:
- Removed the hardcoded Hugging Face token from source.
- Split model kwargs from processor kwargs.
- Made the freeze logic honor the config flags instead of always freezing the encoders.
- Added mask upsampling so predicted masks line up with the input image size.

### 5. Lightweight regression coverage was added for the new masked-item path

Updated file:
- `/root/rest_model/tests/test_item_processing.py`

## Remaining issues

### 1. Stage 1 still prints raw Qwen JSON to stdout during validation/inference

Evidence:
- The completed joint training trial emitted repeated `--- Qwen Raw Output ---` blocks during validation.

Impact:
- This is noisy but not a functional blocker. It makes logs harder to read and pollutes command output during longer runs.

### 2. `latest_metrics.json` ends on the last event, not the best validation summary

Evidence:
- After the successful run, `/root/rest_model/logs/trial-20260321-full3/joint/latest_metrics.json` ended on the epoch checkpoint event instead of the `eval_epoch` summary.

Impact:
- Consumers that read only `latest_metrics.json` do not get the validation metrics directly. The authoritative metrics are still present in `events.jsonl` and the checkpoint events.

### 3. Heavy model tests are still not suitable as a fast validation layer

Evidence:
- The existing test suite still contains heavyweight model construction paths.
- I did not treat the full suite as a fast correctness signal because that would conflate infrastructure/runtime issues with the refactor itself.

Impact:
- Fast regression protection still depends mostly on the lightweight tests and targeted smoke checks.

## Validation performed

- `python -m compileall /root/rest_model`
  - passed
- `python` manifest-backed smoke check for `load_masked_item_image()`
  - passed, returning a real masked crop from `/root/dataset/_review/dataset/stage3_item_classification.jsonl`
- `pytest -q tests/test_item_processing.py`
  - passed (`2 passed`)
- runtime dependency install for `transformers`, `peft`, `bitsandbytes`, `accelerate`, `huggingface_hub`, `safetensors`, `PyYAML`, `Pillow`, `numpy`, `pytest`
  - completed
- real-batch Stage 3 loss smoke check after the loss fix
  - passed, reducing the masked episodic loss to a normal finite scale (`~1.33` on the checked batch)
- real-batch SAM3 `predict()` smoke check on a validation batch after the dtype fix
  - passed
- full joint training trial
  - command: `python train_joint.py run.name=trial-20260321-full3 logging.wandb=false data.num_workers=0 hardware.device=auto hardware.compile=false hardware.load_in_4bit=true joint.training.epochs=1 joint.training.batch_size=1 joint.training.grad_accum_steps=1 joint.eval.interval=1 stage3.episode.n_way=3 stage3.episode.k_shot=1 stage3.episode.query_per_class=1 stage3.eval.n_way=3 stage3.eval.k_shot=1 stage3.eval.episodes=3`
  - status: completed
  - elapsed: `315.595s`
  - validation summary:
    - `val/loss_total = 4.5622`
    - `val/stage1_recall@0.5 = 0.7368`
    - `val/stage2_mIoU = 0.5697`
    - `val/stage3_acc = 0.6000`

## Overall assessment

The code now matches the requested three-stage contract and has been exercised in a full end-to-end joint training run with validation. The previous hard blockers were fixed in this pass: Stage 3 loss masking/label smoothing is numerically stable, and SAM3 evaluation no longer fails on mixed `bfloat16`/`float32` attention inputs. The remaining issues are smaller operational issues rather than blockers.
