#!/bin/sh
# Starting directory
echo "Working directory: $(pwd)"

# Downloading data
python data_setup.py classification

# List of experiments to perform for the next run
# Cross Val Baseline
python runner.py --task_type classification_crossval --models mobilenetv3 efficientnet shufflenet --mode train --epochs 60 --lr 1e-4 --batch 32 --workers 8 --patience 10 --label cross_val_baseline
python runner.py --task_type classification_crossval --models cnn --mode train --epochs 70 --lr 1e-3 --batch 32 --workers 8 --patience 10 --label cross_val_baseline

# Cross Val RoI
python runner.py --task_type classification_crossval --models mobilenetv3 efficientnet shufflenet --mode train --epochs 60 --lr 1e-4 --batch 32 --workers 8 --patience 10 --roi --label cross_val_roi
python runner.py --task_type classification_crossval --models cnn --mode train --epochs 70 --lr 1e-3 --batch 32 --workers 8 --patience 10 --roi --label cross_val_roi

# # Cross Val Weighted Loss
python runner.py --task_type classification_crossval --models mobilenetv3 efficientnet shufflenet --mode train --epochs 60 --lr 1e-4 --batch 32 --workers 8 --patience 10 --weighted_loss --label cross_val_weighted_loss
python runner.py --task_type classification_crossval --models cnn --mode train --epochs 70 --lr 1e-3 --batch 32 --workers 8 --patience 10 --weighted_loss --label cross_val_weighted_loss

# # Cross Val Weighted Sampling
python runner.py --task_type classification_crossval --models mobilenetv3 efficientnet shufflenet --mode train --epochs 60 --lr 1e-4 --batch 32 --workers 8 --patience 10 --weighted_sampling --label cross_val_weighted_sampling
python runner.py --task_type classification_crossval --models cnn --mode train --epochs 70 --lr 1e-3 --batch 32 --workers 8 --patience 10 --weighted_sampling --label cross_val_weighted_sampling

# Cross Val RoI with Weighted Loss
echo "Cross Val RoI with Weighted Loss"
python runner.py --task_type classification_crossval --models mobilenetv3 efficientnet shufflenet --mode train --epochs 60 --lr 1e-4 --batch 32 --workers 8 --patience 10 --roi --weighted_loss --label cross_val_roi_weighted_loss
python runner.py --task_type classification_crossval --models cnn --mode train --epochs 70 --lr 1e-3 --batch 32 --workers 8 --patience 10 --roi --weighted_loss --label cross_val_roi_weighted_loss

# Cross Val RoI with Weighted Sampling
echo "Cross Val RoI with Weighted Sampling"
python runner.py --task_type classification_crossval --models mobilenetv3 efficientnet shufflenet --mode train --epochs 60 --lr 1e-4 --batch 32 --workers 8 --patience 10 --roi --weighted_sampling --label cross_val_roi_weighted_sampling
python runner.py --task_type classification_crossval --models cnn --mode train --epochs 70 --lr 1e-3 --batch 32 --workers 8 --patience 10 --roi --weighted_sampling --label cross_val_roi_weighted_sampling

# Cross Val RoI with Both
echo "Cross Val RoI with Both Weighted Loss and Sampling"
python runner.py --task_type classification_crossval --models mobilenetv3 efficientnet shufflenet --mode train --epochs 60 --lr 1e-4 --batch 32 --workers 8 --patience 10 --roi --weighted_loss --weighted_sampling --label cross_val_roi_weighted_both
python runner.py --task_type classification_crossval --models cnn --mode train --epochs 70 --lr 1e-3 --batch 32 --workers 8 --patience 10 --roi --weighted_loss  --weighted_sampling --label cross_val_roi_weighted_both

# Siamese
# echo "Siamese Few-Shot with ShuffleNet backend"
# python runner.py --task_type siamese --models shufflenet --mode train --epochs 60 --lr 1e04 --batch 32 --workers 8 --patience 10 --weighted_sampling --label siamese_roi_weighted_sampling