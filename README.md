# Dissertation: Model Repo
Model architecture, code and weights for MSc Dissertation Project: "Mobile-Optimised Deep Learning Framework for Interpretable Detection of Fungal Keratitis via Lateral Flow Device Imagery".

### What's in this repository?
This repository contains the full codebase for the training and evaluation of deep learning models developed for the dissertation project. It includes scripts for data processing, model training, and performance benchmarking, along with the saved results and model weights.

## Setup
Create a fresh python virtual environment and run `install.py` to install all the required dependencies. Alternatively, you can manually install the required dependencies using the `requirements.txt` file.

## Runs
Python bench supports training, validation and evaluation for different methodologies: Conventional classification (fine-tuning and training from scratch), Siamese Few-Shot training, Knowledge Distillation. The training regime and models used are selected based on command-line arguments. Here are a few examples of how to run the bench (assuming the data is already prepared): 

1. **To run a classification cross-validation with no additional mechanisms (baseline):**
```bash
python runner.py --task_type classification_crossval --models mobilenetv3 efficientnet shufflenet --mode train --epochs 60 --lr 1e-4 --batch 32 --workers 8 --patience 10 --label cross_val_baseline
```

2. **To run a traditional classification to save model:**
```bash
python runner.py --task_type classification --models efficientnet --mode train --epochs 60 --lr 1e-4 --batch 32 --workers 8 --patience 10 --roi --weighted_sampling --label efficientnet_roi_weighted_sampling
```

3. **To run an evaluation on a saved model:**
```bash
python runner.py --task_type classification --models efficientnet --mode evaluate --batch 32 --workers 8 --patience 10 --roi --weighted_sampling --label efficientnet_roi_weighted_sampling --file efficientnet_fold1_2025-06-16_15-05-49.pth
```

Since the test bench switches the type of training, regime used and models trained based on the command arguments passed, please refer to the code and `scripts/next.sh` for more parameters and running commands.

## Saved Results
All the saved results of experiments are stored at `src/results/`, which are processed and visualised in different jupyter notebooks under `experiements/`.

## Saved Weights
The weights of models of training runs are stored at `src/weights/`, under the corresponding experiment labels.
