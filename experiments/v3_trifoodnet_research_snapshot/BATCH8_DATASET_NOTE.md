# Batch8 Dataset Note

This note documents the image package used for the V3 MVP and the public sample
subset included in this repository.

## Source Package

The main V3 input package was `batch_results_v8_500` from the local
`v3_3stage_mvp` workspace. The corresponding batch summary recorded:

- `500` total images
- `467` successful processed cases
- `33` failed cases
- average runtime about `1.5s` per image

Each successful case was stored as a folder containing:

- `original.jpg`
- `visualization.jpg`
- `results.json`
- `crops/`
- `masks/`

## Public Samples Included Here

Representative examples are included under:

- `../../assets/v3/batch8_samples/Cluster_0_frame_frame_025403_00/`
- `../../assets/v3/batch8_samples/Cluster_105_frame_frame_000879_00/`
- `../../assets/v3/batch8_samples/Cluster_161_frame_frame_091147_00/`
- `../../assets/v3/batch8_samples/Cluster_34_frame_frame_111575_00/`

Each public sample folder contains:

- `original.jpg`
- `visualization.jpg`
- `results.json`

## Why Only A Sample Set

The full batch package is much larger than is appropriate for a clean public Git
repository. The sample set is included to make the V3 README and review path
concrete while keeping the repo lightweight enough to browse and clone.
