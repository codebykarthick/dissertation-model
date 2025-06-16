#!/bin/sh
COMMIT_MESSAGE="$1"
export YOLO_CONFIG_DIR=/tmp/ultralytics_config

# Starting directory
echo "Working directory: $(pwd)"

# Downloading data
python data_setup.py classification

# List of experiments to perform for the next run
# # Cross Val Baseline
# echo "Running cross validation with baseline setup"
# python runner.py --task_type classification_crossval --models mobilenetv3 efficientnet shufflenet --mode train --epochs 60 --lr 1e-4 --batch 32 --workers 8 --patience 10 --label cross_val_baseline
# python runner.py --task_type classification_crossval --models cnn --mode train --epochs 70 --lr 1e-3 --batch 32 --workers 8 --patience 10 --label cross_val_baseline

# # Cross Val RoI
# echo "Running cross validation with RoI"
# python runner.py --task_type classification_crossval --models mobilenetv3 efficientnet shufflenet --mode train --epochs 60 --lr 1e-4 --batch 32 --workers 8 --patience 10 --roi --label cross_val_roi
# python runner.py --task_type classification_crossval --models cnn --mode train --epochs 70 --lr 1e-3 --batch 32 --workers 8 --patience 10 --roi --label cross_val_roi

# # # Cross Val Weighted Loss
# echo "Running cross validation with Weighted loss"
# python runner.py --task_type classification_crossval --models mobilenetv3 efficientnet shufflenet --mode train --epochs 60 --lr 1e-4 --batch 32 --workers 8 --patience 10 --weighted_loss --label cross_val_weighted_loss
# python runner.py --task_type classification_crossval --models cnn --mode train --epochs 70 --lr 1e-3 --batch 32 --workers 8 --patience 10 --weighted_loss --label cross_val_weighted_loss

# # # Cross Val Weighted Sampling
# echo "Running cross validation with weighted sampling"
# python runner.py --task_type classification_crossval --models mobilenetv3 efficientnet shufflenet --mode train --epochs 60 --lr 1e-4 --batch 32 --workers 8 --patience 10 --weighted_sampling --label cross_val_weighted_sampling
# python runner.py --task_type classification_crossval --models cnn --mode train --epochs 70 --lr 1e-3 --batch 32 --workers 8 --patience 10 --weighted_sampling --label cross_val_weighted_sampling

# # Cross Val RoI with Weighted Loss
# echo "Running Cross Val RoI with Weighted Loss"
# python runner.py --task_type classification_crossval --models mobilenetv3 efficientnet shufflenet --mode train --epochs 60 --lr 1e-4 --batch 32 --workers 8 --patience 10 --roi --weighted_loss --label cross_val_roi_weighted_loss
# python runner.py --task_type classification_crossval --models cnn --mode train --epochs 70 --lr 1e-3 --batch 32 --workers 8 --patience 10 --roi --weighted_loss --label cross_val_roi_weighted_loss

# # Cross Val RoI with Weighted Sampling
# echo "Running Cross Val RoI with Weighted Sampling"
# python runner.py --task_type classification_crossval --models mobilenetv3 efficientnet shufflenet --mode train --epochs 60 --lr 1e-4 --batch 32 --workers 8 --patience 10 --roi --weighted_sampling --label cross_val_roi_weighted_sampling
# python runner.py --task_type classification_crossval --models cnn --mode train --epochs 70 --lr 1e-3 --batch 32 --workers 8 --patience 10 --roi --weighted_sampling --label cross_val_roi_weighted_sampling

# # Cross Val RoI with Both
# echo "Running Cross Val RoI with Both Weighted Loss and Sampling"
# python runner.py --task_type classification_crossval --models mobilenetv3 efficientnet shufflenet --mode train --epochs 60 --lr 1e-4 --batch 32 --workers 8 --patience 10 --roi --weighted_loss --weighted_sampling --label cross_val_roi_weighted_both
# python runner.py --task_type classification_crossval --models cnn --mode train --epochs 70 --lr 1e-3 --batch 32 --workers 8 --patience 10 --roi --weighted_loss  --weighted_sampling --label cross_val_roi_weighted_both

# # Save an instance of the model.
# echo "Running training on Test set for Normal fine tuning - EfficientNet"
# python runner.py --task_type classification --models efficientnet --mode train --epochs 60 --lr 1e-4 --batch 32 --workers 8 --patience 10 --roi --weighted_sampling --label efficientnet_roi_weighted_sampling
# echo "Running classification on shufflenet with RoI and Weighted sampling to save the model."
# python runner.py --task_type classification --models shufflenet --mode train --epochs 60 --lr 1e-4 --batch 32 --workers 8 --patience 10 --roi --weighted_sampling --label shufflenet_roi_weighted_sampling

# Siamese
echo "Running Siamese Few-Shot"
python runner.py --task_type siamese --models mobilenetv3 efficientnet shufflenet --mode train --epochs 60 --lr 1e-4 --batch 32 --workers 8 --patience 10 --roi --weighted_sampling --label siamese_fewshot

# # Evaluation on test set
# echo "Running evaluation on Test set for EfficientNet fine tuning"
# python runner.py --task_type classification --models efficientnet --mode evaluate --batch 32 --workers 8 --patience 10 --roi --weighted_sampling --label efficientnet_roi_weighted_sampling --file efficientnet_fold1_2025-06-16_13-21-33.pth
# echo "Running evaluation on Test set for Normal fine tuning"
# python runner.py --task_type classification --models shufflenet --mode evaluate --batch 32 --workers 8 --patience 10 --roi --weighted_sampling --label shufflenet_roi_weighted_sampling --file shufflenet_fold1_2025-06-14_11-53-58.pth

# echo "Running evaluation on Test set for siamese few shot tuning"
# python runner.py --task_type siamese --models shufflenet --mode evaluate --batch 32 --workers 8 --patience 10 --roi --weighted_sampling --label siamese_shufflenet --file shufflenet_fold1_2025-06-14_12-44-42.pth
# python runner.py --task_type siamese --models mobilenetv3 --mode evaluate --batch 32 --workers 8 --patience 10 --roi --weighted_sampling --label siamese_fewshot --file 
# python runner.py --task_type siamese --models efficientnet --mode evaluate --batch 32 --workers 8 --patience 10 --roi --weighted_sampling --label siamese_fewshot --file 

### SHUTDOWN POD
# Check if Commit message is empty
if [ -z "$COMMIT_MESSAGE" ]; then
  echo "Error: Commit message is required."
  exit 1
fi

# Commit the results to the repo before shutting down
echo "Committing results with message: $COMMIT_MESSAGE"

# Cd to src/ directory.
cd ..
git add .
git commit -m "$COMMIT_MESSAGE"
git push origin main

# Shutdown Call to terminate the pod
curl -X DELETE "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}" \
     -H "Authorization: Bearer ${API_KEY}"