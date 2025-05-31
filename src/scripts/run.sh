## Download the dataset
python data_setup.py classification
python data_setup.py yolo


## YOLO Fine Tuning
python roi_runner.py --mode train --batch 24

# CLASSIFICATION - No cross validation
## Without RoI
### Training scripts
#### No Oversampling and No loss weighting
nohup python runner.py --task_type classification --models mobilenetv3 efficientnet shufflenet --mode train --epochs 60 --lr 1e-4 --batch 32 --workers 8 --patience 10 > out.log 2>&1 &
python runner.py --models cnn --mode train --epochs 70 --lr 1e-3 --batch 32 --workers 8 --patience 10 

#### With loss weighting
python runner.py --models mobilenetv3 efficientnet shufflenet --mode train --epochs 60 --lr 1e-4 --batch 32 --workers 8 --patience 10  --weighted_loss
python runner.py --models cnn --mode train --epochs 70 --lr 1e-3 --batch 32 --workers 8 --patience 10  --weighted_loss

#### With Weighted sampling
python runner.py --models mobilenetv3 efficientnet shufflenet --mode train --epochs 60 --lr 1e-4 --batch 32 --workers 8 --patience 10  --weighted_sampling
python runner.py --models cnn --mode train --epochs 70 --lr 1e-3 --batch 32 --workers 8 --patience 10  --weighted_sampling

### Testing scripts
python runner.py --models mobilenetv3 --mode evaluate --file mobilenetv3_2025-05-20_09-38-24_val_0.3869.pth
python runner.py --models cnn --mode evaluate --file cnn_2025-05-20_09-41-03_val_0.3092.pth

## With RoI
### Training Scripts

# CLASSIFICATION - With Cross Validation
nohup python runner.py --task_type classification_crossval --models mobilenetv3 --mode train --epochs 60 --lr 1e-4 --batch 32 --workers 8 --patience 10 > out.log 2>&1 &

# SIAMESE