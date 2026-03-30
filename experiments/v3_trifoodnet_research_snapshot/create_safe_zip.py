# =============================================================================
# FILE: create_safe_zip.py
# CATEGORY: UTIL
# PURPOSE: Packaging helper for earlier delivery bundles.
# DEPENDENCIES: None
# USED BY: None
# KEY CLASSES/FUNCTIONS: main
# LAST MODIFIED: 2026-03-21T10:08:54+00:00
# SNAPSHOT NOTES: no major inline issues detected during snapshot generation
# =============================================================================
import os
import shutil
import zipfile
from pathlib import Path

def main():
    bundle_dir = Path(r"d:\downloads\rest_model\dist\trifoodnet_bundle")
    zip_path = Path(r"d:\downloads\rest_model\trifoodnet_deploy.zip")

    print(f"Preparing bundle directory at {bundle_dir}...")
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)

    print("Copying repository files...")
    repo_dir = Path(r"d:\downloads\rest_model")
    shutil.copytree(
        repo_dir, 
        bundle_dir / "rest_model", 
        ignore=shutil.ignore_patterns(".git", "__pycache__", "checkpoints", "logs", "outputs", "*.zip", "dist")
    )

    print("Copying dataset...")
    dataset_dir = Path(r"d:\downloads\Restaurant_dataset\repo\experiments\v3_3stage_mvp\batch_results_v8_500")
    shutil.copytree(dataset_dir, bundle_dir / "dataset")

    print("Updating configuration file paths for portable zip execution...")
    config_path = bundle_dir / "rest_model" / "master_config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = f.read()

        config = config.replace(
            'batch_root:    "d:/downloads/Restaurant_dataset/repo/experiments/v3_3stage_mvp/batch_results_v8_500"',
            'batch_root:    "../dataset"'
        )
        # Handle backslashes in path config as well just in case
        config = config.replace(
            r'batch_root:    "d:\downloads\Restaurant_dataset\repo\experiments\v3_3stage_mvp\batch_results_v8_500"',
            'batch_root:    "../dataset"'
        )

        with open(config_path, "w", encoding="utf-8") as f:
            f.write(config)

    print(f"Creating safely compressed cross-platform ZIP at {zip_path}...")
    if zip_path.exists():
         zip_path.unlink()
         
    # Generate Zip using standard ZipFile, which avoids shutil corruption bugs on large Windows trees
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(bundle_dir):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(bundle_dir)
                zipf.write(file_path, arcname)

    print("\n✅ New Zip Successfully Created!")
    print(f"📦 Location: {zip_path}")

if __name__ == "__main__":
    main()
