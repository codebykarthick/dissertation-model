#!/bin/bash
COMMIT_MESSAGE="$1"
export YOLO_CONFIG_DIR=/tmp/ultralytics_config

# Starting directory
echo "Working directory: $(pwd)"

# Downloading data
if [ ! -d "dataset" ]; then
  echo "dataset/ not found. Running data_setup..."
  python data_setup.py classification
else
  echo "dataset/ already exists. Skipping data_setup."
fi

# List of experiments to perform for the next run
# # Cross Val Baseline
# echo "Running cross validation with baseline setup"
# python runner.py --task_type classification_crossval --models mobilenetv3 efficientnet shufflenet --mode train --epochs 60 --lr 1e-4 --batch 32 --workers 8 --patience 10 --label cross_val_baseline
# python runner.py --task_type classification_crossval --models tinyvit --mode train --epochs 60 --lr 1e-4 --batch 32 --workers 8 --patience 10 --label cross_val_baseline
# python runner.py --task_type classification_crossval --models cnn --mode train --epochs 70 --lr 1e-3 --batch 32 --workers 8 --patience 10 --label cross_val_baseline

# # Cross Val RoI
# echo "Running cross validation with RoI"
# python runner.py --task_type classification_crossval --models mobilenetv3 efficientnet shufflenet --mode train --epochs 60 --lr 1e-4 --batch 32 --workers 8 --patience 10 --roi --label cross_val_roi
# python runner.py --task_type classification_crossval --models tinyvit --mode train --epochs 60 --lr 1e-4 --batch 32 --workers 8 --patience 10 --roi --label cross_val_roi
# python runner.py --task_type classification_crossval --models cnn --mode train --epochs 70 --lr 1e-3 --batch 32 --workers 8 --patience 10 --roi --label cross_val_roi

# # # Cross Val Weighted Loss
# echo "Running cross validation with Weighted loss"
# python runner.py --task_type classification_crossval --models mobilenetv3 efficientnet shufflenet --mode train --epochs 60 --lr 1e-4 --batch 32 --workers 8 --patience 10 --weighted_loss --label cross_val_weighted_loss
# python runner.py --task_type classification_crossval --models tinyvit --mode train --epochs 60 --lr 1e-4 --batch 32 --workers 8 --patience 10 --weighted_loss --label cross_val_weighted_loss
# python runner.py --task_type classification_crossval --models cnn --mode train --epochs 70 --lr 1e-3 --batch 32 --workers 8 --patience 10 --weighted_loss --label cross_val_weighted_loss

# # # Cross Val Weighted Sampling
# echo "Running cross validation with weighted sampling"
# python runner.py --task_type classification_crossval --models mobilenetv3 efficientnet shufflenet --mode train --epochs 60 --lr 1e-4 --batch 32 --workers 8 --patience 10 --weighted_sampling --label cross_val_weighted_sampling
# python runner.py --task_type classification_crossval --models tinyvit --mode train --epochs 60 --lr 1e-4 --batch 32 --workers 8 --patience 10 --weighted_sampling --label cross_val_weighted_sampling
# python runner.py --task_type classification_crossval --models cnn --mode train --epochs 70 --lr 1e-3 --batch 32 --workers 8 --patience 10 --weighted_sampling --label cross_val_weighted_sampling

# # Cross Val RoI with Weighted Loss
# echo "Running Cross Val RoI with Weighted Loss"
# python runner.py --task_type classification_crossval --models mobilenetv3 efficientnet shufflenet --mode train --epochs 60 --lr 1e-4 --batch 32 --workers 8 --patience 10 --roi --weighted_loss --label cross_val_roi_weighted_loss
# python runner.py --task_type classification_crossval --models tinyvit --mode train --epochs 60 --lr 1e-4 --batch 32 --workers 8 --patience 10 --roi --weighted_loss --label cross_val_roi_weighted_loss
# python runner.py --task_type classification_crossval --models cnn --mode train --epochs 70 --lr 1e-3 --batch 32 --workers 8 --patience 10 --roi --weighted_loss --label cross_val_roi_weighted_loss

# # Cross Val RoI with Weighted Sampling
# echo "Running Cross Val RoI with Weighted Sampling"
# python runner.py --task_type classification_crossval --models mobilenetv3 efficientnet shufflenet --mode train --epochs 60 --lr 1e-4 --batch 32 --workers 8 --patience 10 --roi --weighted_sampling --label cross_val_roi_weighted_sampling
# python runner.py --task_type classification_crossval --models tinyvit --mode train --epochs 60 --lr 1e-4 --batch 32 --workers 8 --patience 10 --roi --weighted_sampling --label cross_val_roi_weighted_sampling
# python runner.py --task_type classification_crossval --models cnn --mode train --epochs 70 --lr 1e-3 --batch 32 --workers 8 --patience 10 --roi --weighted_sampling --label cross_val_roi_weighted_sampling

# # Cross Val RoI with Both
# echo "Running Cross Val RoI with Both Weighted Loss and Sampling"
# python runner.py --task_type classification_crossval --models mobilenetv3 efficientnet shufflenet --mode train --epochs 60 --lr 1e-4 --batch 32 --workers 8 --patience 10 --roi --weighted_loss --weighted_sampling --label cross_val_roi_weighted_both
# python runner.py --task_type classification_crossval --models tinyvit --mode train --epochs 60 --lr 1e-4 --batch 32 --workers 8 --patience 10 --roi --weighted_loss --weighted_sampling --label cross_val_roi_weighted_both
# python runner.py --task_type classification_crossval --models cnn --mode train --epochs 70 --lr 1e-3 --batch 32 --workers 8 --patience 10 --roi --weighted_loss  --weighted_sampling --label cross_val_roi_weighted_both

# Save an instance of the model.
# echo "Running training on Test set for Normal fine tuning - EfficientNet"
# python runner.py --task_type classification --models efficientnet --mode train --epochs 60 --lr 1e-4 --batch 32 --workers 8 --patience 10 --roi --weighted_sampling --label efficientnet_roi_weighted_sampling
# echo "Running classification on shufflenet with RoI and Weighted sampling to save the model."
# python runner.py --task_type classification --models shufflenet --mode train --epochs 60 --lr 1e-4 --batch 32 --workers 8 --patience 10 --roi --weighted_sampling --label shufflenet_roi_weighted_sampling

# Siamese
# echo "Running Siamese Few-Shot"
# python runner.py --task_type siamese --models mobilenetv3 efficientnet shufflenet --mode train --epochs 60 --lr 1e-4 --batch 32 --workers 8 --patience 10 --roi --weighted_sampling --label siamese_fewshot

# # Evaluation on test set
# echo "Running evaluation on Test set for EfficientNet fine tuning"
# python runner.py --task_type classification --models efficientnet --mode evaluate --batch 32 --workers 8 --patience 10 --roi --weighted_sampling --label efficientnet_roi_weighted_sampling --file efficientnet_fold1_2025-06-16_15-05-49.pth
# echo "Running evaluation on Test set for Normal fine tuning"
# python runner.py --task_type classification --models shufflenet --mode evaluate --batch 32 --workers 8 --patience 10 --roi --weighted_sampling --label shufflenet_roi_weighted_sampling --file shufflenet_fold1_2025-06-16_15-09-39.pth

# echo "Running evaluation on Test set for siamese few shot tuning"
# python runner.py --task_type siamese --models shufflenet --mode evaluate --batch 32 --workers 8 --patience 10 --roi --weighted_sampling --label siamese_fewshot --file shufflenet_fold1_2025-06-16_14-34-10.pth
# python runner.py --task_type siamese --models mobilenetv3 --mode evaluate --batch 32 --workers 8 --patience 10 --roi --weighted_sampling --label siamese_fewshot --file mobilenetv3_fold1_2025-06-16_14-25-06.pth
# python runner.py --task_type siamese --models efficientnet --mode evaluate --batch 32 --workers 8 --patience 10 --roi --weighted_sampling --label siamese_fewshot --file efficientnet_fold1_2025-06-16_14-30-18.pth

# # Knowledge Distillation
# kd_run() {
#   local T="$1"
#   local A="$2"
#   echo "Running KD with Temperature: ${T} and Alpha: ${A}"
#   local dest="./weights/knowledge_distillation_T_${T}_al_${A}"
#   mkdir -p "${dest}/efficientnet"
#   mkdir -p "${dest}/shufflenet"
#   cp "./weights/efficientnet_roi_weighted_sampling/efficientnet/efficientnet_fold1_2025-06-16_15-05-49.pth" "${dest}/efficientnet/"
#   cp "./weights/shufflenet_roi_weighted_sampling/shufflenet/shufflenet_fold1_2025-06-16_15-09-39.pth" "${dest}/shufflenet/"
#   python runner.py --task_type distillation --models student --mode train --batch 32 --workers 8 --patience 10 --roi --weighted_sampling --label "knowledge_distillation_T_${T}_al_${A}" --teacher1 efficientnet_fold1_2025-06-16_15-05-49.pth --teacher2 shufflenet_fold1_2025-06-16_15-09-39.pth --temperature ${T} --alpha ${A}
# }

# # Training
# echo "Running knowledge distillation training grid search"
# Grid search over Temperature and Alpha values
# temperatures=(1.0 2.0 5.0 10.0)
# alphas=(0.2 0.5 0.8)
# for T in "${temperatures[@]}"; do
#   for A in "${alphas[@]}"; do
#     echo "Starting KD run with T=$T, alpha=$A"
#     kd_run "$T" "$A"
#   done
# done

# # Evaluation
# echo "Running knowledge distillation evaluation"
# python runner.py --task_type distillation --models student --mode evaluate --batch 32 --workers 8 --patience 10 --roi --weighted_sampling --label "knowledge_distillation_T_2.0_al_0.8" --teacher1 efficientnet_fold1_2025-06-16_15-05-49.pth --teacher2 shufflenet_fold1_2025-06-16_15-09-39.pth --temperature 2.0 --alpha 0.8 --file student_fold1_2025-06-25_09-49-39.pth

# # Grad CAM Interpretation
# echo "Running grad cam interpretations on models"
# python runner.py --task_type gradcam --models student --mode evaluate --batch 32 --workers 8 --patience 10 --roi --weighted_sampling --label knowledge_distillation_T_2.0_al_0.8 --file student_fold1_2025-06-25_09-49-39.pth
# python runner.py --task_type gradcam --models efficientnet --mode evaluate --batch 32 --workers 8 --patience 10 --roi --weighted_sampling --label efficientnet_roi_weighted_sampling --file efficientnet_fold1_2025-06-16_15-05-49.pth
# python runner.py --task_type gradcam --models shufflenet --mode evaluate --batch 32 --workers 8 --patience 10 --roi --weighted_sampling --label shufflenet_roi_weighted_sampling --file shufflenet_fold1_2025-06-16_15-09-39.pth

# # MC Dropout
# echo "Running MCDropout inferrence on models"
# python runner.py --task_type mcdropout --models efficientnet --mode evaluate --batch 32 --workers 8 --patience 10 --roi --weighted_sampling --label efficientnet_roi_weighted_sampling --file efficientnet_fold1_2025-06-16_15-05-49.pth
# python runner.py --task_type mcdropout --models shufflenet --mode evaluate --batch 32 --workers 8 --patience 10 --roi --weighted_sampling --label shufflenet_roi_weighted_sampling --file shufflenet_fold1_2025-06-16_15-09-39.pth

# Converting models to mobile format 
echo "Converting Full models into torchscript format"
python runner.py --task_type mobile --models efficientnet --mode export --label efficientnet_roi_weighted_sampling --file efficientnet_fold1_2025-06-16_15-05-49.pth
python runner.py --task_type mobile --models shufflenet --mode export --label shufflenet_roi_weighted_sampling --file shufflenet_fold1_2025-06-16_15-09-39.pth

# Evaluating mobile format performance
echo "Evaluating mobile format models"
python runner.py --task_type mobile --models efficientnet --mode evaluate --label efficientnet_roi_weighted_sampling --file efficientnet_mobile.pt
python runner.py --task_type mobile --models shufflenet --mode evaluate --label shufflenet_roi_weighted_sampling --file shufflenet_mobile.pt

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