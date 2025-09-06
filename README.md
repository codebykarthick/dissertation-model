# Dissertation: Model Repo
Model architecture, code and weights for MSc Dissertation Project: "Mobile-Optimised Deep Learning Framework for Interpretable Detection of Fungal Keratitis via Lateral Flow Device Imagery".

### Important Info
The original dataset used in this dissertation involved private medical data and has been stripped to preserve privacy. Users can insert their own data into `src/data/` instead.

### What's in this repository?
This repository contains the full codebase for the training and evaluation of deep learning models developed for the dissertation project. It includes scripts for data processing, model training, and performance benchmarking, along with the saved results and model weights.

## Organisation
* `experiments/` - Contains additional visualisations and tests run to infer data saved in runs.
* `grad-cam/` - Stores results of Grad-CAM runs of models to see their region of focus on images.
* `models/` - Model architecture code for fine-tuning and other methodologies.
* `results/` - Stored metrics of test-set evaluation - AUC-PR, AUC-ROC, F1.
* `scripts/` - Scripts used for headless, cloud training.
* `util/` - Utility functions needed for runs.
* `weights/` - Saved model weights of best runs for each methodology per architecture.

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

## 📊 Results

### Held-out Test Set Performance
- **EfficientNet-B0**
  - Accuracy: **93.3%**
  - Precision: **1.00**
  - Recall: **0.667**
  - F1: **0.80**
  - Validation Loss: **0.47** (threshold = 0.56)

- **ShuffleNetV2**
  - Accuracy: **86.7%**
  - Precision: **1.00**
  - Recall: **0.333**
  - F1: **0.50**
  - Validation Loss: **0.51** (threshold = 0.47)

---

### AUC Metrics (threshold-independent)
- **EfficientNet-B0**: PR-AUC **0.880**, ROC-AUC **0.940**  
- **ShuffleNetV2**: PR-AUC **0.650**, ROC-AUC **0.760**

---

### Mobile Inference (iPhone 11 Benchmarks)
- All models ran **fully offline** with no network calls.  
- **ShuffleNetV2** showed the lowest latency and memory usage (< 500 MB).  
- **Ensemble inference** was slower but still within device constraints.

---

### Interpretability
- **Grad-CAM** visualisations confirmed EfficientNet focused consistently on the diagnostic lines of the LFDs.  
- ShuffleNet produced less precise attention maps.

