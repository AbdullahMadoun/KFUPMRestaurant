## 2026-03-21

- Cleared old experiment artifacts to free disk space. Remaining retained run: `trial-20260321-full3`.
- Confirmed current supported-class contract keeps 18 classes and removes only `Bashamil`, `salad`, and `tabboulah`.
- Verified the current split no longer collapses to 4 classes, but rare-class allocation still needs refinement so train keeps stronger support while dev/test still receive harder mixed images.
- Identified a Stage 3 compile wiring issue: `torch.compile()` was being applied to an alias instead of the transformer actually used by the PictSure forward path.
- In progress now:
  - strengthen the split heuristic for rare classes
  - wire Stage use a3 compilation to the live PictSure transformer
  - launch a fresh early-stopped convergence run after the code path is stable

## 2026-03-21 Split Update

- Stage 3 compile wiring has been fixed so compilation targets the live PictSure transformer used in forward/inference.
- Splitter objective changed again to prioritize per-split class coverage first, then rare-class train support, then hard-example balancing.
- Current coverage check:
  - supported classes: `18`
  - removed classes: `Bashamil`, `salad`, `tabboulah`
  - train images: `93`
  - dev images: `12`
  - test images: `11`
- Verified 3-image classes are now distributed one image per split:
  - `Fasoolia`: train `1`, dev `1`, test `1`
  - `Koshari`: train `1`, dev `1`, test `1`
  - `Muttabal`: train `1`, dev `1`, test `1`
- Remaining work:
  - run the full convergence training trial on the updated split
  - inspect generated train/dev graphs and dev visualization outputs

## 2026-03-21 Runtime Issue

- First `trial-20260321-converge4` launch failed during startup/training handoff.
- Root cause: properly wiring `torch.compile()` to the live Stage 3 PictSure transformer exposed a CUDA-graphs incompatibility with the PEFT LoRA wrapper.
- Action taken:
  - keep the Stage 3 compile fix in place
  - skip Stage 3 compile when the transformer is PEFT-wrapped
  - relaunch the full convergence run on the same split logic

## 2026-03-21 Convergence Run

- Active run: `trial-20260321-converge5`
- Stage 3 compile is now skipped intentionally when PEFT LoRA is active.
- Startup checks:
  - train images: `93`
  - dev images: `12`
  - test images: `11`
  - Stage 3 reference support images: `64`
- First optimizer step completed with finite losses:
  - `train_step/loss_stage1 = 1.0259`
  - `train_step/loss_stage2 = 3.3658`
  - `train_step/loss_stage3 = 5.7275`
  - `train_step/loss_total = 11.3000`
  - `train_step/stage3_acc = 0.8000`
- Current focus:
  - monitor epoch 1 through the first dev evaluation
  - confirm no `inf`/`nan` regressions
  - verify report and visualization artifacts after checkpointing/eval

## 2026-03-21 Training Loop Fix

- While monitoring `trial-20260321-converge5`, I found a remainder-gradient bug in the joint trainer.
- With `93` train batches and `grad_accum_steps=8`, the last `5` batches of each epoch were not producing an optimizer step.
- The final partial accumulation window was also being scaled as if it contained `8` batches instead of the true remainder.
- Fix applied:
  - detect the true accumulation target for each accumulation window
  - step the optimizer on the final short window
  - log `train_step/accum_target` for visibility
- Result:
  - the current run was stopped
  - the next relaunch will use the corrected trainer

## 2026-03-21 Scheduler Fix

- After fixing remainder accumulation, I found the LR scheduler was still using floor-divided optimizer-step counts.
- On this setup, `93` batches with `grad_accum_steps=8` should yield `12` optimizer steps per epoch, not `11`.
- Fix applied:
  - scheduler total steps now use `ceil(batches_per_epoch / grad_accum_steps)`
  - run metadata now logs `optimizer/optimizer_steps_per_epoch`
- Result:
  - the `trial-20260321-converge6` launch was stopped early
  - the next relaunch will use the corrected optimizer-step schedule

## 2026-03-21 Corrected Long Run

- Active run: `trial-20260321-converge7`
- Verified corrected startup metadata:
  - `optimizer/batches_per_epoch = 93`
  - `optimizer/optimizer_steps_per_epoch = 12`
  - `optimizer/total_steps = 720`
  - `optimizer/warmup_steps = 108`
- First optimizer step completed successfully after the trainer/scheduler fixes:
  - `train_step/loss_stage1 = 1.1596`
  - `train_step/loss_stage2 = 6.6203`
  - `train_step/loss_stage3 = 6.0564`
  - `train_step/loss_total = 13.5543`
  - `train_step/accum_target = 8`
- Current status:
  - no compile crash
  - no dropped remainder batches in the trainer
  - no scheduler step-count mismatch
  - run is continuing toward the first dev evaluation and early-stopped convergence

## 2026-03-21 Teacher-Forcing Alignment

- Confirmed Stage 3 inference was already receiving single-item masked crops, not the full dish image.
- Found the remaining mismatch in training:
  - Stage 3 support came from episodic reference images
  - Stage 3 queries were still detached episode crops instead of the current batch image items
- Fix applied:
  - current-batch item masks and labels are now used to build Stage 3 query inputs during collation
  - support episodes are duplicated per current query item so the PictSure transformer still gets proper ICL context
- Batch smoke check after the patch:
  - `support_images = (3, 25, 3, 224, 224)`
  - `query_images = (3, 1, 3, 224, 224)`
  - `query_labels = (3,)`
- Additional training stability fix:
  - Stage 2 BCE/Dice loss weights from config were previously ignored
  - they are now wired into `SAM3Segmenter`

## 2026-03-21 PictSure Training Answer

- Genuine answer to "what is being trained in PictSure":
  - the masked single-item crops are the image inputs
  - the support labels are turned into label vectors by PictSure's `y_projection`
  - the query item gets the same image encoder path but an empty label slot
  - the actual in-context classification is done by the trainable transformer over `[image_embedding ; label_embedding]` tokens
- In the current refactor, the trainable Stage 3 path is:
  - PictSure image encoder / embedding module
  - `x_projection`
  - `y_projection`
  - the transformer encoder
  - the final classifier head `fc`
- We also attach LoRA to the live PictSure transformer, so the few-shot reasoning path itself is trainable rather than frozen.
- This means Stage 3 is not acting like a plain nearest-neighbor classifier. It is learning the ICL mapping from support item embeddings plus support label embeddings to the query class.

## 2026-03-21 Stage 3 Label-Path Fixes

- Found the first hard Stage 3 instability bug:
  - current-image query items were being paired with random support episodes
  - in many cases the query class was not present in support
  - that produced pathological Stage 3 CE spikes around `50+`
- Fix applied:
  - each current query item now gets a support episode anchored on that item's class
  - support size remains fixed at `n_way * k_shot = 25`
- Found the second hard Stage 3 instability bug:
  - query labels were built from `classes.json` enumeration
  - manifest `class_id` values use a different indexing scheme
  - this meant the model could see the correct crop but still be trained against the wrong target index
- Fix applied:
  - class-name to id mapping is now derived from manifest `class_id` values
  - train/inference/reporting code now uses the same class-id aligned name index
- Verification after both fixes:
  - sampled query items checked: `69`
  - query items missing from support: `0`
  - support size per query episode: `25`
- Next step:
  - relaunch the monitored 3-epoch stability run
  - keep tuning until we get 3 epochs with consistent loss reduction

## 2026-03-21 Stability Run `trial-20260321-stability2`

- The 3-epoch objective trend is now moving in the right direction:
  - epoch 1 `dev/loss_total = 13.133476`
  - epoch 2 `dev/loss_total = 13.045835`
  - epoch 3 `dev/loss_total = 13.037923`
- This is the first clean run after the Stage 3 support/query fixes where the dev loss decreased at every epoch instead of exploding.
- Important clarification on the dev accuracy question:
  - `dev/stage3_episode_acc = 0.318182` is the teacher-forced episodic classifier metric
  - `dev/stage3_acc = 0.0` is the live end-to-end Qwen -> SAM -> PictSure metric
  - `dev/stage3_matched_acc = 0.0` is also zero, so this is not just a denominator artifact
  - the metric code is doing exact matched-label counting, so the zero appears to be real rather than a calculation bug
- Current diagnosis:
  - Stage 3 training objective is no longer collapsed
  - end-to-end inference is still failing because the live segmentation/classification path is not aligning with ground truth
  - the strongest signal is `dev/stage2_mIoU = 0.0`, which suggests the SAM inference/NMS path is still the main blocker

## 2026-03-21 Validation Pass

- Added repeatable validation tooling:
  - [validate_pipeline_contracts.py](/root/rest_model/validate_pipeline_contracts.py)
  - [test_pipeline_contracts.py](/root/rest_model/tests/test_pipeline_contracts.py)
- Fast validation status:
  - `pytest -q tests/test_pipeline_contracts.py tests/test_item_processing.py`
  - result: `5 passed`
- Contract validation report:
  - saved at [validation_report.json](/root/rest_model/validation_report.json)
  - class-id alignment passes
  - split contract passes
  - Stage 3 support/query episode contract passes on all `164` checked train queries
- Runtime validation findings:
  - the old `dev/stage3_acc = 0.0` was not a pure metric bug
  - disabling post-SAM NMS and fixing CPU/GPU mask device mismatches changed the full-dev live metric from `0.0` to `0.04545`
  - runtime audit on 5 dev images with NMS off produced masks on `4/5` images
  - one audited image still returned `0` predicted items
  - Stage 2 masks are flowing now, but `stage2_mIoU` remains `0.0`, so the mask content is still poor
- Additional inference fixes applied:
  - post-SAM mask NMS can now be disabled cleanly
  - `master_config.yaml` now defaults to `stage2.nms.iou_threshold = 0.0` and `score_threshold = 0.0`
  - live mask tensors are normalized onto CPU before masked cropping / metrics
  - Stage 3 inference no longer feeds the transformer one giant 18-class context by default; it now retrieves a 5-way candidate support episode first
- Current remaining failure:
  - even after retrieval-based Stage 3 inference, the audited runtime predictions still collapse toward a small subset of classes instead of matching the GT labels reliably

## 2026-03-21 Stability Run `trial-20260321-stability3`

- The rerun with the full Stage 3 checkpoint save/load fix completed successfully.
- Dev loss still decreased cleanly across all 3 epochs:
  - epoch 1 `dev/loss_total = 12.816538`
  - epoch 2 `dev/loss_total = 12.743077`
  - epoch 3 `dev/loss_total = 12.732777`
- The raw logged dev inference metrics were still misleadingly poor at the time of the run:
  - `dev/stage1_recall@0.5 = 0.772727`
  - `dev/stage2_mIoU = 0.0`
  - `dev/stage3_acc = 0.045455`
- I treated that as a debugging signal, not as the final truth about the model.

## 2026-03-21 Rigorous Data Audit

- Added stronger data-contract checks into [validate_pipeline_contracts.py](/root/rest_model/validate_pipeline_contracts.py):
  - full annotation geometry / mask / support-episode checks across train, dev, test
  - Stage 3 support-capacity reporting
  - Stage 3 crop loading audit over all retained rows
  - teacher-forced Stage 2 and Stage 3 runtime audits
- Added regression coverage in:
  - [test_pipeline_contracts.py](/root/rest_model/tests/test_pipeline_contracts.py)
  - [test_stage2_sam.py](/root/rest_model/tests/test_stage2_sam.py)
- Current local test status:
  - `pytest -q tests/test_stage2_sam.py tests/test_pipeline_contracts.py tests/test_item_processing.py`
  - result: `9 passed`
- Full data-only validation report:
  - saved at [validation_report_data_only.json](/root/rest_model/validation_report_data_only.json)
- Structural findings from the retained dataset:
  - train: `93` images, `164` labeled items, `55` multi-item images
  - dev: `12` images, `22` labeled items, `6` multi-item images
  - test: `11` images, `15` labeled items, `4` multi-item images
  - all `201` retained Stage 3 rows have `mask_path`
  - blank masked crops found: `0`
- Remaining data pressure:
  - these train classes still have fewer than `k_shot=5` support items and therefore force support duplication during Stage 3 episodes:
    `Aseeda_brown`, `Aseeda_white`, `Chicken_kebab`, `Fasoolia`, `Koshari`, `Meat_kebab`, `Muttabal`, `cake`, `hummus`, `soup`

## 2026-03-21 Stage 2 Decoding Fix

- Found the real Stage 2 runtime bug:
  - HF `Sam3Model` returns DETR-style query outputs, not one mask per prompt box
  - raw output for one GT box was:
    - `pred_masks = (1, 200, 288, 288)`
    - `pred_logits = (1, 200)`
    - `pred_boxes = (1, 200, 4)`
  - the old wrapper treated all `200` queries as prompt outputs and supervised the first query against the GT mask
- Fix applied in [stage2_sam.py](/root/rest_model/stage2_sam.py):
  - scale normalized `pred_boxes` into image space
  - greedily match SAM3 query boxes back to the input grounding boxes
  - return one aligned mask/logit per prompt instead of `200` unrelated query masks
- Added unit coverage so that this query-to-prompt decoding does not regress.

## 2026-03-21 Post-Fix Runtime Readout

- After the Stage 2 decoding fix, the same saved `trial-20260321-stability3` checkpoint behaves very differently in post-hoc validation:
  - teacher-forced Stage 2 dev mIoU: `0.657829`
  - live end-to-end dev Stage 2 mIoU: `0.488746`
  - teacher-forced Stage 3 dev accuracy: `0.227273`
  - live end-to-end dev Stage 3 accuracy: `0.090909`
  - live end-to-end dev matched Stage 3 accuracy: `0.117647`
- Interpretation:
  - Stage 2 is no longer the dominant failure under teacher forcing
  - Stage 1 -> Stage 2 handoff is now usable enough to propagate masks into Stage 3
  - the main remaining bottleneck is Stage 3 classification quality
- Additional Stage 3 diagnostic:
  - retrieval hit rate for the true class being present in the 5-way support subset on dev GT crops is `0.863636`
  - so retrieval is not the primary failure anymore; the classifier itself is still underperforming

## 2026-03-21 Corrected Retrain `trial-20260321-stability4`

- Launched a fresh 3-epoch stability rerun after fixing SAM3 query decoding.
- Early train behavior is immediately better than `stability3`:
  - epoch 1 step-10 train interval:
    - `loss_stage2 = 0.930393`
    - `loss_stage3 = 5.583096`
    - `loss_total = 10.652817`
- First dev evaluation is also materially better:
  - epoch 1 `dev/loss_total = 11.142423`
  - epoch 1 `dev/stage1_recall@0.5 = 0.772727`
  - epoch 1 `dev/stage2_mIoU = 0.489727`
  - epoch 1 `dev/stage3_acc = 0.181818`
  - epoch 1 `dev/stage3_episode_acc = 0.409091`
  - epoch 1 `dev/stage3_matched_acc = 0.235294`
- Comparison against `trial-20260321-stability3` epoch 1:
  - `dev/loss_total`: `12.816538 -> 11.142423`
  - `dev/stage2_mIoU`: `0.0 -> 0.489727`
  - `dev/stage3_acc`: `0.045455 -> 0.181818`
- Current decision:
  - keep the corrected retrain running
  - do not change the Stage 3 loss yet unless later epochs flatten or regress

## 2026-03-21 `trial-20260321-stability4` Outcome

- The corrected 3-epoch rerun stayed numerically stable and kept improving the total dev objective:
  - epoch 1 `dev/loss_total = 11.142423`
  - epoch 2 `dev/loss_total = 11.075720`
  - epoch 3 `dev/loss_total = 11.062845`
- Stage 2 stayed fixed at the improved level across all epochs:
  - epoch 1 `dev/stage2_mIoU = 0.489727`
  - epoch 2 `dev/stage2_mIoU = 0.490860`
  - epoch 3 `dev/stage2_mIoU = 0.490686`
- Stage 3 did not follow the same improvement:
  - epoch 1 `dev/stage3_acc = 0.181818`
  - epoch 2 `dev/stage3_acc = 0.136364`
  - epoch 3 `dev/stage3_acc = 0.090909`
  - epoch 1/2/3 `dev/stage3_episode_acc = 0.409091`
- Diagnosis:
  - the SAM3 fix solved the Stage 2 bottleneck
  - the remaining issue is now the Stage 3 training objective / optimization behavior
  - dev loss is still going down while Stage 3 accuracy goes down, so the current Stage 3 objective is not aligned with the metric we need
- Action taken for the next rerun:
  - changed `stage3.loss.name` from `balanced_softmax` to `cross_entropy`
  - changed `stage3.training.label_smoothing` from `0.1` to `0.0`
  - rationale: the previous Stage 3 setup appears to be over-biasing or over-smoothing on this small long-tail support library

## 2026-03-21 Long-Run Launch Settings

- Promoted the corrected short-run settings into the main joint defaults in [master_config.yaml](/root/rest_model/master_config.yaml):
  - `joint.training.epochs = 40`
  - `joint.training.warmup_ratio = 0.05`
  - `joint.training.early_stopping.monitor = dev/loss_total`
  - `joint.training.early_stopping.mode = min`
  - `joint.training.early_stopping.patience = 8`
  - `joint.training.early_stopping.min_epochs = 8`
  - `joint.curriculum.gt_boxes_epochs = 3`
- Rationale:
  - the old defaults were still optimized for the pre-fix pipeline and held GT boxes for the full run
  - after the SAM3 query-decoding fix, `dev/loss_total` became the most reliable convergence signal
  - a short GT-box curriculum better matches the real inference path and reduces teacher-forcing mismatch
- Next action:
  - launch a full 40-epoch joint training run with report generation and dev visualizations enabled
  - collect final train/dev metrics, plots, checkpoints, and report paths under `/root/rest_model`

## 2026-03-21 Long Run `trial-20260321-full40-crossent1` In Flight

- The full corrected run launched successfully with:
  - `run.name = trial-20260321-full40-crossent1`
  - `logging.wandb = false`
  - `optimizer/total_steps = 480`
  - `optimizer/warmup_steps = 24`
  - `joint.curriculum.gt_boxes_epochs = 3`
- Startup checks passed:
  - Qwen loaded cleanly
  - SAM3 loaded with the prompt-aligned query decoding fix active
  - pretrained PictSure loaded and LoRA attached to the live transformer
- Early train behavior is finite and substantially better than the pre-fix runs:
  - epoch 1 step-10 interval:
    - `train_step/loss_total = 3.604950`
    - `train_step/loss_stage3 = 0.884741`
    - `train_step/stage3_acc = 0.816667`
- The monitored dev objective has improved for 3 consecutive epochs:
  - epoch 1 `dev/loss_total = 4.195167`
  - epoch 2 `dev/loss_total = 4.092097`
  - epoch 3 `dev/loss_total = 4.000943`
- Supporting dev metrics over the same window:
  - epoch 1 `dev/stage2_mIoU = 0.488465`, `dev/stage3_acc = 0.090909`
  - epoch 2 `dev/stage2_mIoU = 0.488981`, `dev/stage3_acc = 0.045455`
  - epoch 3 `dev/stage2_mIoU = 0.485531`, `dev/stage3_acc = 0.090909`
- Interpretation so far:
  - the pipeline is stable enough to continue the long run
  - Stage 2 remains healthy after the SAM3 fix
  - Stage 3 accuracy is still noisy on the tiny dev set, but the teacher-forced objective is now improving consistently
  - the next key check is epoch 4+, when the run leaves the GT-box bootstrap and uses predicted boxes end-to-end

## 2026-03-21 Teacher-Forcing Curriculum Revision

- Stopped `trial-20260321-full40-crossent1` after the user requested that teacher forcing remain active for most of training.
- Reason for the change:
  - the previous schedule switched from GT boxes to predicted boxes after only `3` epochs
  - once that happened, train-step latency jumped from about `1s` to about `32s`, and the handoff was too abrupt to count as a stability-first curriculum
- Implemented a scheduled teacher-forcing curriculum in [train_joint.py](/root/rest_model/train_joint.py):
  - epochs `1-28`: teacher-forcing probability `1.0`
  - epochs `29-40`: linearly decay teacher forcing from `1.0` to `0.25`
  - teacher forcing is now sampled per train step instead of a single hard epoch cutoff
  - logged per-step and per-epoch so the curriculum is visible in the JSONL metrics
- Updated defaults in [master_config.yaml](/root/rest_model/master_config.yaml):
  - `joint.curriculum.teacher_forcing.sustain_epochs = 28`
  - `joint.curriculum.teacher_forcing.transition_epochs = 12`
  - `joint.curriculum.teacher_forcing.start_prob = 1.0`
  - `joint.curriculum.teacher_forcing.end_prob = 0.25`
  - `joint.training.early_stopping.min_epochs = 28`
- Verified the new schedule directly:
  - epoch `1` -> `1.0`
  - epoch `28` -> `1.0`
  - epoch `29` -> `0.9375`
  - epoch `34` -> `0.625`
  - epoch `40` -> `0.25`
- Next action:
  - relaunch the full 40-epoch run with the revised stability-first curriculum and collect the final train/dev reports from that run only

## 2026-03-21 Constant Teacher Forcing `0.8`

- The staged teacher-forcing curriculum was superseded by a stricter user request:
  - use a fixed teacher-forcing probability of `0.8`
  - run a full `40` epochs
- Updated [master_config.yaml](/root/rest_model/master_config.yaml):
  - `joint.curriculum.teacher_forcing.sustain_epochs = 40`
  - `joint.curriculum.teacher_forcing.transition_epochs = 0`
  - `joint.curriculum.teacher_forcing.start_prob = 0.8`
  - `joint.curriculum.teacher_forcing.end_prob = 0.8`
  - `joint.training.early_stopping.enabled = false`
- Verified the trainer wiring directly:
  - early stopping is disabled
  - teacher forcing probability resolves to `0.8` at epochs `1`, `20`, and `40`
- Next action:
  - launch a fresh 40-epoch retrain using the constant `0.8` teacher-forcing probability and report results from that run only

## 2026-03-21 Full Retrain `trial-20260321-full40-tf08-1` Running

- Relaunched the full retrain under:
  - `run.name = trial-20260321-full40-tf08-1`
  - `run.notes = 40-epoch full retrain with constant 0.8 teacher forcing probability`
  - `logging.wandb = false`
- Verified in the live log:
  - `curriculum/teacher_forcing_start_prob = 0.8`
  - `curriculum/teacher_forcing_end_prob = 0.8`
  - `curriculum/teacher_forcing_transition_epochs = 0`
  - `train/teacher_forcing_prob = 0.8` at epoch 1
  - `joint.training.early_stopping.enabled = false`
- Early runtime status:
  - startup is clean for Qwen, SAM3, and PictSure
  - losses are finite
  - GT-box usage is now sampled batch-by-batch against the fixed `0.8` teacher-forcing probability
- Important runtime note:
  - this configuration is materially slower than the previous pure-teacher-forced epochs because any non-teacher-forced batch must run live Qwen grounding inside training
  - the run is still proceeding correctly; the longer wall-clock time is expected from the requested curriculum

## 2026-03-21 Pure Teacher Forcing Decision

- After observing the `0.8` run, the real bottleneck was confirmed:
  - slowdown was caused by live Qwen grounding inside non-teacher-forced training batches
  - it was not a VRAM-capacity issue; peak GPU memory stayed around `26 GB` on a `32 GB` card
- User decision applied:
  - switch to pure teacher forcing for training
- Updated [master_config.yaml](/root/rest_model/master_config.yaml):
  - `joint.curriculum.teacher_forcing.start_prob = 1.0`
  - `joint.curriculum.teacher_forcing.end_prob = 1.0`
  - full `40` epochs still required
  - early stopping remains disabled so the run completes the full schedule
- Important behavior:
  - train batches now always use GT boxes
  - dev evaluation still uses full inference (`Qwen -> SAM3 -> PictSure`) so we keep a truthful held-out signal
- Next action:
  - relaunch the definitive 40-epoch pure-teacher-forcing run and use that run only for final reporting

## 2026-03-21 Dev Image Source Audit

- Investigated whether dev inference was accidentally using GT boxes or pre-made annotations.
- Findings:
  - the actual dev evaluator still runs `pipeline.run(...)`, so the intended path remains live `Qwen -> SAM3 -> PictSure`
  - however, the exported image sources are corrupted at the file-path level:
    - all `118/118` manifest `image_path` entries point to `original.jpg`
    - all `118/118` of those `original.jpg` files are invalid non-image pointer files
    - all `118/118` point to `dummy.jpg`
    - the loader has therefore been silently using `visualization.jpg` for every image
- Code fixes applied:
  - [dataset_integration.py](/root/rest_model/dataset_integration.py) now resolves image provenance explicitly and exposes `image_source_kind` / `resolved_image_path`
  - [validate_pipeline_contracts.py](/root/rest_model/validate_pipeline_contracts.py) now audits image-source provenance in the validation report
  - [visualize_val_predictions.py](/root/rest_model/visualize_val_predictions.py) no longer defaults to GT-box prompting; live Stage 1 inference is now the default and GT-box mode is explicit via `--use-gt-boxes`
  - [post_training_artifacts.py](/root/rest_model/post_training_artifacts.py) now records image provenance in dev visualization outputs
- Interpretation:
  - this is a real data-export corruption issue
  - it is not the same as the main dev evaluator using GT boxes by default
  - any visualization or report generated from the current export should now disclose whether it came from `direct_image`, `pointer_resolved`, or `visualization_fallback`
  - after confirming from `/root/dataset/pipeline.log` that these `visualization.jpg` files were saved pipeline visualizations, not clean originals, the full-image loader was tightened so Stage 1 / Stage 2 paths no longer silently accept them

## 2026-03-21 Detailed Data Correctness Audit

- Wrote a full current-state audit to [data_correctness_report.json](/root/rest_model/data_correctness_report.json).
- Current split counts:
  - train: `93` images, `164` labeled items, `18` classes
  - dev: `12` images, `22` labeled items, `17` classes
  - test: `11` images, `15` labeled items, `13` classes
- Full-image source integrity:
  - `118 / 118` `original.jpg` files are invalid
  - `118 / 118` point to `dummy.jpg`
  - `118 / 118` cluster folders contain `visualization.jpg` and `results.json`
  - verdict: full-image Stage 1 / Stage 2 training is not safe on the current export
- Item-level asset integrity:
  - Stage 3 rows: `205`
  - loadable crops: `205 / 205`
  - loadable masks: `205 / 205`
  - blank masks: `1`
  - verdict: Stage 3 item assets are mostly intact, but the set is not perfectly clean
- User-facing split folders were also adjusted:
  - removed `visualization.jpg` symlinks from `/root/dataset/_review/dataset/splits/*`
  - current split folders expose crops, masks, results, and metadata only
- Required reupload contents for a correct end-to-end run:
  - clean original full images for every `image_id`
  - a stable `image_id -> original image file` mapping
  - no generated visualization overlays used as full-image sources

## 2026-03-21 `images.zip` Audit And Pointer Repair

- Audited [images.zip](/root/images.zip):
  - it contains `images/<image_id>/original.jpg` pointer files, crops, masks, results, and visualizations
  - it does **not** contain the actual raw `Sampled_Images_All/*.jpg` binaries
  - sample pointer content now correctly references paths like:
    - `/workspace/KFUPMRestaurant/Sampled_Images_All/Cluster_124_frame_frame_043685_00.jpg`
- Applied the useful part of that upload to the active dataset:
  - patched `118 / 118` current `original.jpg` pointer files away from `dummy.jpg`
  - current patch report: [images_zip_pointer_patch_report.json](/root/rest_model/images_zip_pointer_patch_report.json)
- Materialized split-to-original pointer maps:
  - [train_original_pointers.json](/root/dataset/_review/dataset/splits/train_original_pointers.json)
  - [dev_original_pointers.json](/root/dataset/_review/dataset/splits/dev_original_pointers.json)
  - [test_original_pointers.json](/root/dataset/_review/dataset/splits/test_original_pointers.json)
- Remaining blocker:
  - the actual unoverlaid raw full-image files are still not present locally
  - full-image Stage 1 / Stage 2 training and dev inference remain blocked until those binaries are uploaded or mounted

## 2026-03-21 Raw Image Recovery And Final Data Repair

- Pulled the raw image source directly from the public branch:
  - repository: `AbdullahMadoun/KFUPMRestaurant`
  - branch: `3-stage-MVP`
  - sparse path: `Sampled_Images_All`
- Synced validated raw images into:
  - [Sampled_Images_All](/root/rest_model/Sampled_Images_All)
  - recovered count: `809` real image files
- Manual visual spot-check confirmed the recovered raw image for `Cluster_124_frame_frame_043685_00` is clean and differs from the old overlaid `visualization.jpg`.
- Strict validation on the reuploaded `_review` package plus recovered raw images showed:
  - `118 / 118` full images resolved without visualization fallback
  - `0` full images were identical to visualization outputs
  - one remaining defect: a blank reviewed mask for `Cluster_122_frame_frame_038337_00`
- Applied final cleanup:
  - discarded the single corrupted image `Cluster_122_frame_frame_038337_00`
  - synced the reuploaded review assets into `/root/dataset/_review`
  - rebuilt split folders so `original.jpg` is a working symlink to the validated raw image
  - kept visualization assets excluded from split folders
- Final strict audit result written to [data_correctness_report.json](/root/rest_model/data_correctness_report.json):
  - `image_rows = 117`
  - `stage3_rows = 204`
  - `resolved_ok = 117`
  - `resolved_fail_count = 0`
  - `identical_to_visualization_count = 0`
  - `crop_ok = 204`
  - `mask_ok = 204`
  - `blank_masks = 0`
  - verdict:
    - `full_image_training_safe = true`
    - `stage3_item_training_safe = true`
- Verified split-folder originals open correctly from:
  - `/root/dataset/_review/dataset/splits/train/.../original.jpg`
  - `/root/dataset/_review/dataset/splits/dev/.../original.jpg`
  - `/root/dataset/_review/dataset/splits/test/.../original.jpg`

## 2026-03-21 Clean-Data Training Launch

- Started a fresh joint run on the repaired dataset:
  - `run.name = trial-20260321-cleandata1`
  - `run.notes = 40-epoch retrain after raw-image recovery and corrupted sample removal`
- Startup checks from the live run log:
  - dataset split sizes: train `93`, dev `12`, test `11`
  - total optimizer steps: `480`
  - warmup steps: `24`
  - Stage 3 compile remains intentionally skipped because the PEFT transformer path is not compatible with the current compile setup
  - teacher forcing is active at `1.0`
- First confirmed train steps are finite:
  - step `1`: `loss_total = 2.2760`
  - step `2`: `loss_total = 10.0951`
  - step `3`: `loss_total = 2.9918`
- Current log path:
  - [events.jsonl](/root/rest_model/logs/trial-20260321-cleandata1/joint/events.jsonl)
- Current metrics snapshot:
  - [latest_metrics.json](/root/rest_model/logs/trial-20260321-cleandata1/joint/latest_metrics.json)

## 2026-03-21 Clean-Data Training Monitoring

- Monitored the live run through the epoch `2` dev evaluation.
- Epoch `1` dev metrics:
  - `loss_total = 4.0153`
  - `stage1_recall@0.5 = 0.7727`
  - `stage2_mIoU = 0.5406`
  - `stage3_acc = 0.2727`
  - `stage3_matched_acc = 0.3529`
- Epoch `2` dev metrics:
  - `loss_total = 3.8593`
  - `stage1_recall@0.5 = 0.8182`
  - `stage1_precision@0.5 = 0.7826`
  - `stage2_mIoU = 0.5606`
  - `stage3_acc = 0.3182`
  - `stage3_matched_acc = 0.3889`
  - `stage3_episode_acc = 0.5909`
- Result:
  - epoch `2` improved over epoch `1` on total dev loss, Stage 1 recall, Stage 2 mIoU, and Stage 3 accuracy
  - no non-finite losses observed so far
  - the run produced a new best checkpoint at epoch `2`

- Epoch `3` dev metrics:
  - `loss_total = 3.6890`
  - `stage1_recall@0.5 = 0.8182`
  - `stage1_precision@0.5 = 0.6923`
  - `stage2_mIoU = 0.5453`
  - `stage3_acc = 0.3182`
  - `stage3_matched_acc = 0.3889`
  - `stage3_episode_acc = 0.6364`
- Updated result after epoch `3`:
  - dev total loss improved for three consecutive epochs: `4.0153 -> 3.8593 -> 3.6890`
  - Stage 1 recall held its epoch `2` gain
  - Stage 2 mIoU remained stable above `0.54`
  - Stage 3 accuracy improved from epoch `1` and then held steady through epoch `3`
  - no non-finite losses observed
  - the run produced another new best checkpoint at epoch `3`

- Epochs `4` to `6` dev summary:
  - epoch `4`:
    - `loss_total = 3.5730`
    - `stage1_precision@0.5 = 0.7917`
    - `stage1_recall@0.5 = 0.8636`
    - `stage2_mIoU = 0.5883`
    - `stage3_acc = 0.4091`
    - `stage3_matched_acc = 0.4737`
  - epoch `5`:
    - `loss_total = 3.4924`
    - `stage1_precision@0.5 = 0.7727`
    - `stage1_recall@0.5 = 0.7727`
    - `stage2_mIoU = 0.5197`
    - `stage3_acc = 0.3636`
    - `stage3_matched_acc = 0.4706`
  - epoch `6`:
    - `loss_total = 3.4446`
    - `stage1_precision@0.5 = 0.7391`
    - `stage1_recall@0.5 = 0.7727`
    - `stage2_mIoU = 0.5171`
    - `stage3_acc = 0.3636`
    - `stage3_matched_acc = 0.4706`
    - `stage3_episode_acc = 0.6364`
- Result after epoch `6`:
  - dev total loss continued to improve each epoch through `6`
  - best overall live checkpoint is now epoch `6`
  - Stage 1 and Stage 2 peaked at epoch `4` so the later loss reduction is not uniformly improving every stage metric
  - Stage 3 end-to-end dev accuracy improved materially versus epoch `1` baseline and remains above that baseline
  - no `inf` or `nan` losses observed

## 2026-03-21 Periodic Visualization Hook

- Updated [train_joint.py](/root/rest_model/train_joint.py) so epoch-tagged dev visualization artifacts can be generated after eval every `3` epochs via `logging.visualizations.every_n_epochs`.
- Updated [post_training_artifacts.py](/root/rest_model/post_training_artifacts.py) so prediction artifacts can be rendered without the ground-truth image panel; labels and masks remain visualized, and GT metadata still stays in JSON/HTML.
- Updated [master_config.yaml](/root/rest_model/master_config.yaml) to enable periodic visualization generation every `3` epochs with the full dev set.
- Safeguard added:
  - visualization failures are logged but do not fail training
- Limitation:
  - the currently running `trial-20260321-cleandata1` process was started before this code change, so it will not pick up the new periodic-visualization hook unless relaunched

## 2026-03-21 Epochs 7-9 Monitoring

- Epoch `7` dev metrics:
  - `loss_total = 3.3962`
  - `stage1_precision@0.5 = 0.7391`
  - `stage1_recall@0.5 = 0.7727`
  - `stage2_mIoU = 0.5176`
  - `stage3_acc = 0.3636`
  - `stage3_matched_acc = 0.4706`
- Epoch `8` dev metrics:
  - `loss_total = 3.3631`
  - `stage1_precision@0.5 = 0.7391`
  - `stage1_recall@0.5 = 0.7727`
  - `stage2_mIoU = 0.5180`
  - `stage3_acc = 0.3182`
  - `stage3_matched_acc = 0.4118`
- Epoch `9` dev metrics:
  - `loss_total = 3.3431`
  - `stage1_precision@0.5 = 0.7391`
  - `stage1_recall@0.5 = 0.7727`
  - `stage2_mIoU = 0.5192`
  - `stage3_acc = 0.3636`
  - `stage3_matched_acc = 0.4706`
- Result after epoch `9`:
  - dev total loss kept improving through epochs `7`, `8`, and `9`
  - Stage 1 and Stage 2 are now fairly stable rather than improving sharply
  - Stage 3 remains noisy but is not collapsing
  - no `inf` or `nan` losses observed
  - the best checkpoint was updated through epoch `9`

## 2026-03-21 Epochs 10-12 Monitoring

- Epoch `10` dev metrics:
  - `loss_total = 3.3260`
  - `stage1_precision@0.5 = 0.7391`
  - `stage1_recall@0.5 = 0.7727`
  - `stage2_mIoU = 0.5194`
  - `stage3_acc = 0.4091`
  - `stage3_matched_acc = 0.5294`
- Epoch `11` dev metrics:
  - `loss_total = 3.2950`
  - `stage1_precision@0.5 = 0.7727`
  - `stage1_recall@0.5 = 0.7727`
  - `stage2_mIoU = 0.5180`
  - `stage3_acc = 0.3636`
  - `stage3_matched_acc = 0.4706`
- Epoch `12` dev metrics:
  - `loss_total = 3.2710`
  - `stage1_precision@0.5 = 0.7727`
  - `stage1_recall@0.5 = 0.7727`
  - `stage2_mIoU = 0.5182`
  - `stage3_acc = 0.3636`
  - `stage3_matched_acc = 0.4706`
- Result after epoch `12`:
  - dev total loss kept improving through epochs `10`, `11`, and `12`
  - the strongest point in this block was epoch `10` for Stage 3 end-to-end accuracy
  - Stage 1 and Stage 2 remained broadly stable through epoch `12`
  - no `inf` or `nan` losses observed
  - the best checkpoint continued to update through epoch `12`

- Early warning after epoch `13`:
  - `loss_total` still improved to `3.2502`
  - but Stage 1 recall dropped to `0.7273`, Stage 2 mIoU dropped to `0.4905`, and Stage 3 accuracy dropped to `0.2727`
  - this is the first clear sign that lower total loss is no longer aligning with better end-to-end dev behavior

## 2026-03-21 Epochs 13-15 Monitoring

- Epoch `13` dev metrics:
  - `loss_total = 3.2502`
  - `stage1_precision@0.5 = 0.6957`
  - `stage1_recall@0.5 = 0.7273`
  - `stage2_mIoU = 0.4905`
  - `stage3_acc = 0.2727`
  - `stage3_matched_acc = 0.3750`
- Epoch `14` dev metrics:
  - `loss_total = 3.2269`
  - `stage1_precision@0.5 = 0.8000`
  - `stage1_recall@0.5 = 0.7273`
  - `stage2_mIoU = 0.4981`
  - `stage3_acc = 0.2727`
  - `stage3_matched_acc = 0.3750`
- Epoch `15` dev metrics:
  - `loss_total = 3.2244`
  - `stage1_precision@0.5 = 0.7500`
  - `stage1_recall@0.5 = 0.6818`
  - `stage2_mIoU = 0.4637`
  - `stage3_acc = 0.3182`
  - `stage3_matched_acc = 0.4667`
- Result after epoch `15`:
  - dev total loss continued to decrease, but the end-to-end dev metrics degraded versus the epoch `10` to `12` window
  - Stage 1 recall fell from `0.7727` at epoch `12` to `0.6818` at epoch `15`
  - Stage 2 mIoU fell from `0.5182` at epoch `12` to `0.4637` at epoch `15`
  - Stage 3 accuracy remains noisy and below the epoch `10` peak
  - this now looks like objective misalignment rather than simple optimization failure
  - no `inf` or `nan` losses observed
  - current checkpoint selection is misaligned with deployment quality:
    - [master_config.yaml](/root/rest_model/master_config.yaml) still monitors `dev/loss_total` with `mode=min`
    - so [best](/root/rest_model/checkpoints/trial-20260321-cleandata1/joint/best) is the lowest-loss checkpoint, not the strongest end-to-end checkpoint

## 2026-03-21 Epochs 16-18 Monitoring

- Epoch `16` dev metrics:
  - `loss_total = 3.2254`
  - `stage1_precision@0.5 = 0.8000`
  - `stage1_recall@0.5 = 0.7273`
  - `stage2_mIoU = 0.4777`
  - `stage3_acc = 0.3636`
  - `stage3_matched_acc = 0.5000`
- Epoch `17` dev metrics:
  - `loss_total = 3.2239`
  - `stage1_precision@0.5 = 0.8421`
  - `stage1_recall@0.5 = 0.7273`
  - `stage2_mIoU = 0.4755`
  - `stage3_acc = 0.4091`
  - `stage3_matched_acc = 0.5625`
- Epoch `18` dev metrics:
  - `loss_total = 3.2164`
  - `stage1_precision@0.5 = 0.8421`
  - `stage1_recall@0.5 = 0.7273`
  - `stage2_mIoU = 0.4758`
  - `stage3_acc = 0.3636`
  - `stage3_matched_acc = 0.5000`
- Result after epoch `18`:
  - dev total loss is still drifting downward
  - Stage 1 recall has not recovered beyond `0.7273`
  - Stage 2 mIoU remains below the stronger epoch `10` to `12` range
  - Stage 3 briefly improved at epoch `17`, then slipped back at epoch `18`
  - the model is still numerically stable, but quality remains below the earlier mid-run peak

## 2026-03-21 Epochs 26-28 Monitoring

- Epoch `26` dev metrics:
  - `loss_total = 3.1939`
  - `stage1_precision@0.5 = 0.6800`
  - `stage1_recall@0.5 = 0.7727`
  - `stage2_mIoU = 0.5105`
  - `stage3_acc = 0.3636`
  - `stage3_matched_acc = 0.4706`
  - `latency_total_ms = 5697.9`
- Epoch `27` dev metrics:
  - `loss_total = 3.1959`
  - `stage1_precision@0.5 = 0.6923`
  - `stage1_recall@0.5 = 0.8182`
  - `stage2_mIoU = 0.5333`
  - `stage3_acc = 0.3636`
  - `stage3_matched_acc = 0.4444`
  - `latency_total_ms = 6587.5`
- Epoch `28` dev metrics:
  - `loss_total = 3.1979`
  - `stage1_precision@0.5 = 0.6800`
  - `stage1_recall@0.5 = 0.7727`
  - `stage2_mIoU = 0.5097`
  - `stage3_acc = 0.4091`
  - `stage3_matched_acc = 0.5294`
  - `latency_total_ms = 6518.4`
- Result after epoch `28`:
  - dev loss has flattened around `3.19` rather than continuing to improve materially
  - Stage 1 and Stage 2 recovered relative to the epoch `13` to `18` dip, especially at epoch `27`
  - Stage 3 remains noisy, with the best result in this block at epoch `28`
  - inference latency has drifted up substantially and is now above `5.6s` per dev image

## 2026-03-21 Epochs 29-31 Monitoring

- Epoch `29` dev metrics:
  - `loss_total = 3.2000`
  - `stage1_precision@0.5 = 0.6923`
  - `stage1_recall@0.5 = 0.8182`
  - `stage2_mIoU = 0.5346`
  - `stage3_acc = 0.4091`
  - `stage3_matched_acc = 0.5000`
  - `latency_total_ms = 6493.1`
- Epoch `30` dev metrics:
  - `loss_total = 3.2005`
  - `stage1_precision@0.5 = 0.7600`
  - `stage1_recall@0.5 = 0.8636`
  - `stage2_mIoU = 0.5732`
  - `stage3_acc = 0.5000`
  - `stage3_matched_acc = 0.5789`
  - `latency_total_ms = 6087.6`
- Epoch `31` dev metrics:
  - `loss_total = 3.1967`
  - `stage1_precision@0.5 = 0.7600`
  - `stage1_recall@0.5 = 0.8636`
  - `stage2_mIoU = 0.5732`
  - `stage3_acc = 0.5000`
  - `stage3_matched_acc = 0.5789`
  - `latency_total_ms = 6081.2`
- Result after epoch `31`:
  - end-to-end dev quality improved materially versus the earlier flat middle-run window
  - the strongest recent point is epoch `31`
  - Stage 1 recall and Stage 2 mIoU are both back above their earlier mid-run levels
  - Stage 3 end-to-end accuracy reached `0.50`, the best value seen so far in this run
  - latency is still high, but slightly better than the peak observed around epochs `27` to `29`

## 2026-03-21 Epochs 32-34 Monitoring

- Epoch `32` dev metrics:
  - `loss_total = 3.2026`
  - `stage1_precision@0.5 = 0.7600`
  - `stage1_recall@0.5 = 0.8636`
  - `stage2_mIoU = 0.5733`
  - `stage3_acc = 0.5000`
  - `stage3_matched_acc = 0.5789`
  - `latency_total_ms = 5985.0`
- Epoch `33` dev metrics:
  - `loss_total = 3.2029`
  - `stage1_precision@0.5 = 0.7600`
  - `stage1_recall@0.5 = 0.8636`
  - `stage2_mIoU = 0.5732`
  - `stage3_acc = 0.5000`
  - `stage3_matched_acc = 0.5789`
  - `latency_total_ms = 6134.9`
- Epoch `34` dev metrics:
  - `loss_total = 3.1990`
  - `stage1_precision@0.5 = 0.7600`
  - `stage1_recall@0.5 = 0.8636`
  - `stage2_mIoU = 0.5733`
  - `stage3_acc = 0.5000`
  - `stage3_matched_acc = 0.5789`
  - `latency_total_ms = 6094.4`
- Result after epoch `34`:
  - the run has entered a very stable plateau
  - end-to-end dev metrics are essentially unchanged across epochs `32` to `34`
  - the strongest recent behavior from epochs `30` to `34` is being maintained rather than improved further

## 2026-03-21 Results Report Artifacts

- Generated a full metrics report bundle for `trial-20260321-cleandata1` at [report_metrics](/root/rest_model/outputs/trial-20260321-cleandata1/report_metrics).
- Key report files:
  - [RESULTS_SUMMARY.md](/root/rest_model/outputs/trial-20260321-cleandata1/report_metrics/RESULTS_SUMMARY.md)
  - [index.md](/root/rest_model/outputs/trial-20260321-cleandata1/report_metrics/index.md)
  - [summary.json](/root/rest_model/outputs/trial-20260321-cleandata1/report_metrics/summary.json)
  - [epoch_loss_summary.csv](/root/rest_model/outputs/trial-20260321-cleandata1/report_metrics/epoch_loss_summary.csv)
- Added train-vs-dev epoch overlays:
  - [train_vs_dev_loss_total_by_epoch.svg](/root/rest_model/outputs/trial-20260321-cleandata1/report_metrics/plots/train_vs_dev_loss_total_by_epoch.svg)
  - [train_vs_dev_loss_stage1_by_epoch.svg](/root/rest_model/outputs/trial-20260321-cleandata1/report_metrics/plots/train_vs_dev_loss_stage1_by_epoch.svg)
  - [train_vs_dev_loss_stage2_by_epoch.svg](/root/rest_model/outputs/trial-20260321-cleandata1/report_metrics/plots/train_vs_dev_loss_stage2_by_epoch.svg)
  - [train_vs_dev_loss_stage3_by_epoch.svg](/root/rest_model/outputs/trial-20260321-cleandata1/report_metrics/plots/train_vs_dev_loss_stage3_by_epoch.svg)
- Report snapshot at generation time:
  - completed dev epochs through `40`
  - final log status later settled to `completed` in `run_status.json`

## 2026-03-21 Final Archive Package

- The joint run completed successfully:
  - run status: `completed`
  - finished UTC: `2026-03-21T23:00:56.888899+00:00`
  - elapsed seconds: `7354.59`
- Final dev epoch `40` metrics:
  - `loss_total = 3.2009`
  - `stage1_recall@0.5 = 0.8636`
  - `stage2_mIoU = 0.5734`
  - `stage3_acc = 0.5000`
  - `stage3_matched_acc = 0.5789`
- Final train eval metrics:
  - `train/loss_total = 2.0714`
  - `train/stage1_recall@0.5 = 0.7988`
  - `train/stage2_mIoU = 0.4919`
  - `train/stage3_acc = 0.3537`
- Preserved checkpoint variants in the archive:
  - latest epoch checkpoint: [latest_epoch_040](/root/rest_model/final_archive/trial-20260321-cleandata1/checkpoints/latest_epoch_040)
  - best end-to-end dev checkpoint by `dev/combined`: [best_dev_combined_epoch_038](/root/rest_model/final_archive/trial-20260321-cleandata1/checkpoints/best_dev_combined_epoch_038)
  - existing loss-monitor best checkpoint: [best_loss_monitor_existing](/root/rest_model/final_archive/trial-20260321-cleandata1/checkpoints/best_loss_monitor_existing)
- Final archive folder assembled at [trial-20260321-cleandata1](/root/rest_model/final_archive/trial-20260321-cleandata1)
- Archive includes:
  - curated code snapshot
  - configs and run snapshots
  - finalized logs
  - metric report bundle with train-vs-dev loss visuals
  - dataset manifests and validation audits
  - reference project docs

## 2026-03-21 Repo-Wide Full Wrap

- Built a repo-wide archival bundle at [rest_model_full_wrap_20260321](/root/rest_model/final_archive/rest_model_full_wrap_20260321).
- Built the final zip at [rest_model_full_wrap_20260321.zip](/root/rest_model/final_archive/rest_model_full_wrap_20260321.zip).
- Added repo-wide package docs:
  - [README.md](/root/rest_model/final_archive/rest_model_full_wrap_20260321/README.md)
  - [FINDINGS_AND_RESULTS.md](/root/rest_model/final_archive/rest_model_full_wrap_20260321/FINDINGS_AND_RESULTS.md)
  - [DESIGN_DECISIONS.md](/root/rest_model/final_archive/rest_model_full_wrap_20260321/DESIGN_DECISIONS.md)
  - [REPRODUCIBILITY.md](/root/rest_model/final_archive/rest_model_full_wrap_20260321/REPRODUCIBILITY.md)
  - [CODE_INVENTORY.md](/root/rest_model/final_archive/rest_model_full_wrap_20260321/CODE_INVENTORY.md)
  - [FILE_MANIFEST.md](/root/rest_model/final_archive/rest_model_full_wrap_20260321/FILE_MANIFEST.md)
- Included all 15 trial log directories, the all-trials comparison report, existing per-run reports, the cleaned dataset manifests, raw image pool, and selected final-run checkpoint payloads.
- Added current-run dev inference bundles for:
  - [best_dev_combined_epoch_038](/root/rest_model/final_archive/rest_model_full_wrap_20260321/reports/per_run/trial-20260321-cleandata1/final_wrapup/dev_visualizations/best_dev_combined_epoch_038)
  - [latest_epoch_040](/root/rest_model/final_archive/rest_model_full_wrap_20260321/reports/per_run/trial-20260321-cleandata1/final_wrapup/dev_visualizations/latest_epoch_040)
- Added validation artifacts:
  - [bundle_structure_report.json](/root/rest_model/final_archive/rest_model_full_wrap_20260321/validation/bundle_structure_report.json)
  - [compileall.txt](/root/rest_model/final_archive/rest_model_full_wrap_20260321/validation/compileall.txt)
  - [pytest_smoke.txt](/root/rest_model/final_archive/rest_model_full_wrap_20260321/validation/pytest_smoke.txt)
  - external zip integrity report: [rest_model_full_wrap_20260321_zip_integrity.json](/root/rest_model/final_archive/rest_model_full_wrap_20260321_zip_integrity.json)
- Validation results:
  - bundle structure verifier: `ok=true`
  - packaged source compile check: passed
  - smoke tests: `9 passed`
  - final zip `testzip()`: `ok=true`

## 2026-03-21 Final Cleanup

- Performed a conservative cleanup pass on [rest_model_full_wrap_20260321](/root/rest_model/final_archive/rest_model_full_wrap_20260321) before final delivery.
- Removed unnecessary items from the package only:
  - dropped the redundant loss-monitor checkpoint copy
  - dropped the orphaned `results/dev` carryover
  - trimmed raw images from the full pool down to the `117` images referenced by the reviewed manifest
- Cleanup record:
  - [cleanup_summary.json](/root/rest_model/final_archive/rest_model_full_wrap_20260321/manifests/cleanup_summary.json)
- Post-cleanup package size dropped to about `1.1G` for checkpoints and `35M` for dataset assets inside the wrap.
