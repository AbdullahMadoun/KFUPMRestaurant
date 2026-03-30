# =============================================================================
# FILE: bundle_deployment.py
# CATEGORY: UTIL
# PURPOSE: Deployment bundle helper used by prior archive workflows.
# DEPENDENCIES: None
# USED BY: None
# KEY CLASSES/FUNCTIONS: main
# LAST MODIFIED: 2026-03-21T09:46:36+00:00
# SNAPSHOT NOTES: no major inline issues detected during snapshot generation
# =============================================================================
import os
import shutil
from pathlib import Path

def main():
    bundle_dir = Path(r"d:\downloads\trifoodnet_bundle")
    zip_path = Path(r"d:\downloads\trifoodnet_deployment.zip")

    print(f"Preparing bundle directory at {bundle_dir}...")
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)

    print("Copying repository files...")
    repo_dir = Path(r"d:\downloads\rest_model")
    # Only bring the necessary folders over, discard bulky pycaches or old logs
    shutil.copytree(
        repo_dir, 
        bundle_dir / "rest_model", 
        ignore=shutil.ignore_patterns(".git", "__pycache__", "checkpoints", "logs", "outputs", "*.zip")
    )

    print("Copying dataset...")
    dataset_dir = Path(r"d:\downloads\Restaurant_dataset\repo\experiments\v3_3stage_mvp\batch_results_v8_500")
    shutil.copytree(dataset_dir, bundle_dir / "dataset")

    print("Updating configuration file paths for portable zip execution...")
    config_path = bundle_dir / "rest_model" / "master_config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = f.read()

        # Update the batch_root absolute path to the portable relative path `../dataset`
        config = config.replace(
            'batch_root:    "d:/downloads/Restaurant_dataset/repo/experiments/v3_3stage_mvp/batch_results_v8_500"',
            'batch_root:    "../dataset"'
        )

        with open(config_path, "w", encoding="utf-8") as f:
            f.write(config)

    print(f"Archiving to {zip_path}...")
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", bundle_dir)

    print("\n✅ Bundle successfully created!")
    print(f"📦 Location: {zip_path}")
    print("\nTo deploy:")
    print("1. Transfer `trifoodnet_deployment.zip` to your GPU machine.")
    print("2. Unzip it.")
    print("3. cd into `trifoodnet_bundle/rest_model` and run `python train_joint.py`.")

if __name__ == "__main__":
    main()
