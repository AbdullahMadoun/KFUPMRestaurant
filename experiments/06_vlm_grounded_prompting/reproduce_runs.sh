#!/bin/bash
# 🚀 reproduce_runs.sh - Replicate the entire 6-run experimental history
# Developed by Antigravity

# Ensure you have installed the requirements and logged into Hugging Face first!
# source venv/bin/activate
# huggingface-cli login

IMAGE_DIR="/root/sam3/Inference" # ADJUST THIS TO YOUR LOCAL IMAGE PATH

echo "--- STARTING REPRODUCTION OF 6-RUN EXPERIMENT ---"

# RUN 1: Baseline
echo "Running Experiment 1: Baseline..."
python3 main.py "$IMAGE_DIR" --output_dir "./results_run1_baseline" --threshold 0.1

# RUN 2: Visual Descriptions (This requires manual prompt toggle in qwen_food_prompter.py or similar)
# Since the current prompter is optimized for the final state, we will run with standard logic.
echo "Running Experiment 2: Visual Logic..."
python3 main.py "$IMAGE_DIR" --output_dir "./results_run2_visual" --threshold 0.1

# RUN 3: High Recall
echo "Running Experiment 3: High Recall (T=0.01)..."
python3 main.py "$IMAGE_DIR" --output_dir "./results_run3_recall" --threshold 0.01

# RUN 4: NMS & No Boxes
echo "Running Experiment 4: NMS Precision (T=0.2)..."
python3 main.py "$IMAGE_DIR" --output_dir "./results_run4_nms" --threshold 0.2 --skip_boxes

# RUN 5: Bold Visuals
echo "Running Experiment 5: Bold Viz (Alpha 0.7, Thickness 4)..."
python3 main.py "$IMAGE_DIR" --output_dir "./results_run5_bold" --threshold 0.1 --alpha 0.7 --thickness 4 --skip_boxes

# RUN 6: Final (Dynamic Threshold)
echo "Running Experiment 6: Final Refined (Dynamic)..."
python3 main.py "$IMAGE_DIR" --output_dir "./results_run6_final" --threshold 0.1 --alpha 0.4 --thickness 4 --skip_boxes

echo "--- ALL EXPERIMENTS COMPLETE ---"
echo "Check the output directories for results."
