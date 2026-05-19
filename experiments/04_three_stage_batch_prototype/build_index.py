"""CLI tool to build the FAISS vector index from reference menu images.

Usage:
    python build_index.py \
        --menu_dir /path/to/reference_images/ \
        --schema menu_schema.json \
        --output_index menu.index \
        --output_meta menu_meta.json \
        --device cuda

Expected menu_dir structure:
    chicken/
        img1.jpg, img2.jpg, ...
    rice/
        img1.jpg, ...
    salad/
        img1.jpg, ...

Each subdirectory name must match a key in menu_schema.json.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoProcessor

from vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("pipeline")


def main():
    parser = argparse.ArgumentParser(description="Build FAISS vector index from reference menu images")
    parser.add_argument("--menu_dir", required=True, help="Directory with per-item image subdirectories")
    parser.add_argument("--schema", required=True, help="Path to menu_schema.json")
    parser.add_argument("--output_index", default="menu.index", help="Output FAISS index path")
    parser.add_argument("--output_meta", default="menu_meta.json", help="Output metadata JSON path")
    parser.add_argument("--model", default="google/siglip2-base-patch16-224", help="SigLIP 2 model name")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    args = parser.parse_args()

    # Load schema
    schema_path = Path(args.schema)
    if not schema_path.exists():
        logger.error(f"Schema file not found: {schema_path}")
        return
    schema = json.loads(schema_path.read_text())
    logger.info(f"Loaded schema with {len(schema)} menu items")

    # Verify menu directory
    menu_dir = Path(args.menu_dir)
    if not menu_dir.exists():
        logger.error(f"Menu directory not found: {menu_dir}")
        return

    # Load SigLIP 2 model
    logger.info(f"Loading embedding model: {args.model}")
    start = time.time()
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float16).to(args.device).eval()
    processor = AutoProcessor.from_pretrained(args.model)
    logger.info(f"Model loaded in {time.time() - start:.1f}s")

    # Build index
    store = VectorStore()
    start = time.time()
    store.build(args.menu_dir, schema, model, processor, args.device)
    logger.info(f"Index built in {time.time() - start:.1f}s")

    # Save
    store.save(args.output_index, args.output_meta)
    logger.info(f"Done. Index: {args.output_index} ({store.index.ntotal} vectors), Meta: {args.output_meta}")


if __name__ == "__main__":
    main()
