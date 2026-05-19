# Experiments Index

| Run ID | Date | Config | Key Metric | Value | Checkpoint Path | Notes |
| --- | --- | --- | --- | ---: | --- | --- |
| ⭐ trial-20260321-cleandata1 | 2026-03-21T23:00:56.888899+00:00 | logs/trial-20260321-cleandata1/joint/config_snapshot.json | joint/combined | 1.9375961198969618 | checkpoints/trial-20260321-cleandata1/joint/epoch_038 | 40-epoch retrain after raw-image recovery and corrupted sample removal |
| trial-20260321-converge4 | 2026-03-21T13:34:48.417415+00:00 | logs/trial-20260321-converge4/joint/config_snapshot.json | joint/combined | n/a | n/a | Full convergence run after split coverage and PictSure compile fix |
| trial-20260321-converge5 | unknown | logs/trial-20260321-converge5/joint/config_snapshot.json | joint/combined | n/a | n/a | Full convergence run after split coverage refactor and Stage3 compile skip for PEFT |
| trial-20260321-converge6 | unknown | logs/trial-20260321-converge6/joint/config_snapshot.json | joint/combined | n/a | n/a | Full convergence run after split coverage, PEFT-safe Stage3 compile handling, and gradient remainder fix |
| trial-20260321-converge7 | unknown | logs/trial-20260321-converge7/joint/config_snapshot.json | joint/combined | 0.8181818181818181 | n/a | Full convergence run after split coverage, PEFT-safe Stage3 compile handling, gradient remainder fix, and scheduler step-count fix |
| trial-20260321-full3 | 2026-03-21T11:34:53.338452+00:00 | logs/trial-20260321-full3/joint/config_snapshot.json | joint/combined | 1.9064984766601074 | n/a | First full joint training pass |
| trial-20260321-full40-crossent1 | 2026-03-21T19:43:10.087068+00:00 | logs/trial-20260321-full40-crossent1/joint/config_snapshot.json | joint/combined | 1.3521015969531631 | n/a | 40-epoch full run after SAM3 query decoding fix and Stage3 CE alignment |
| trial-20260321-full40-puretf1 | unknown | logs/trial-20260321-full40-puretf1/joint/config_snapshot.json | joint/combined | 1.3101694644245667 | n/a | 40-epoch full retrain with pure teacher forcing in training and full inference on dev |
| trial-20260321-full40-tf08-1 | 2026-03-21T20:05:04.674863+00:00 | logs/trial-20260321-full40-tf08-1/joint/config_snapshot.json | joint/combined | 1.3970592483157867 | n/a | 40-epoch full retrain with constant 0.8 teacher forcing probability |
| trial-20260321-full40-tfmajority1 | 2026-03-21T19:48:51.696932+00:00 | logs/trial-20260321-full40-tfmajority1/joint/config_snapshot.json | joint/combined | n/a | n/a | 40-epoch full run with scheduled teacher forcing dominant through most epochs |
| trial-20260321-stability1 | unknown | logs/trial-20260321-stability1/joint/config_snapshot.json | joint/combined | n/a | n/a | 3-epoch stability trial after current-item Stage3 queries and Stage2 loss-weight fix |
| trial-20260321-stability2 | unknown | logs/trial-20260321-stability2/joint/config_snapshot.json | joint/combined | 0.7727272727272727 | n/a | 3-epoch stability trial after Stage3 support/query class-id fixes |
| trial-20260321-stability3 | 2026-03-21T14:52:44.166690+00:00 | logs/trial-20260321-stability3/joint/config_snapshot.json | joint/combined | 0.8181818181818181 | n/a | 3-epoch stability rerun after Stage3 full-checkpoint fix and no-NMS defaults |
| trial-20260321-stability4 | unknown | logs/trial-20260321-stability4/joint/config_snapshot.json | joint/combined | 1.4442723853831299 | n/a | 3-epoch stability rerun after Stage2 SAM3 query decoding fix |
| trial-20260321-stability5 | 2026-03-21T15:33:54.982606+00:00 | logs/trial-20260321-stability5/joint/config_snapshot.json | joint/combined | 1.4450006238541302 | n/a | 3-epoch stability rerun after Stage3 loss alignment fix |
