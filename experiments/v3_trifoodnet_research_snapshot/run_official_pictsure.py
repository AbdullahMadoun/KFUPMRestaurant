# =============================================================================
# FILE: run_official_pictsure.py
# CATEGORY: INFER
# PURPOSE: CLI wrapper for the upstream official PictSure checkpoints.
# DEPENDENCIES: pictsure_official.py
# USED BY: None
# KEY CLASSES/FUNCTIONS: build_parser, main
# LAST MODIFIED: 2026-03-21T07:10:30+00:00
# SNAPSHOT NOTES: no major inline issues detected during snapshot generation
# =============================================================================
"""
CLI entrypoint for the official pretrained PictSure models on Hugging Face.

Example:
    python run_official_pictsure.py ^
        --image data/example_crop.jpg ^
        --reference-library data/reference_library ^
        --model-id pictsure/pictsure-vit ^
        --top-k 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from pictsure_official import (
    OfficialPictSureClassifier,
    load_reference_library_from_dir,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Path to the query crop image.")
    parser.add_argument(
        "--reference-library",
        required=True,
        help="Directory with one subfolder per class containing reference images.",
    )
    parser.add_argument(
        "--model-id",
        default="pictsure/pictsure-vit",
        help="Official Hugging Face model id.",
    )
    parser.add_argument("--device", default="auto", help="cpu, cuda, mps, or auto.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of predictions to print.")
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Enable torch.compile when supported and beneficial.",
    )
    return parser


def main():
    args = build_parser().parse_args()

    classifier = OfficialPictSureClassifier(
        model_id=args.model_id,
        device=args.device,
        use_torch_compile=args.torch_compile,
    )
    classifier.build_reference_library(
        load_reference_library_from_dir(args.reference_library)
    )

    with Image.open(Path(args.image)) as query_image:
        predictions = classifier.classify(query_image.convert("RGB"), top_k=args.top_k)

    print(f"Model: {args.model_id}")
    print(f"Query: {args.image}")
    for rank, (label, score) in enumerate(predictions, start=1):
        print(f"{rank:>2}. {label:<24} {score:.4f}")


if __name__ == "__main__":
    main()
