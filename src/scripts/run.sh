# Download the dataset
python data_setup.py classification
python data_setup.py yolo

# Training scripts - Without RoI, No Oversampling and No loss weighting
python runner.py --models mobilenetv3 efficientnet --mode train --epochs 60 --lr 1e-4 --batch 32 --workers 8 --patience 10 
python runner.py --models cnn --mode train --epochs 70 --lr 1e-3 --batch 32 --workers 8 --patience 10 

# Training scripts - Without RoI, with loss weighting
python runner.py --models mobilenetv3 efficientnet --mode train --epochs 60 --lr 1e-4 --batch 32 --workers 8 --patience 10  --weighted_loss
python runner.py --models cnn --mode train --epochs 70 --lr 1e-3 --batch 32 --workers 8 --patience 10  --weighted_loss

# Training scripts - Without RoI, with Weighted sampling
python runner.py --models mobilenetv3 efficientnet --mode train --epochs 60 --lr 1e-4 --batch 32 --workers 8 --patience 10  --weighted_sampling
python runner.py --models cnn --mode train --epochs 70 --lr 1e-3 --batch 32 --workers 8 --patience 10  --weighted_sampling

# Testing scripts - Without RoI
python runner.py --models mobilenetv3 --mode evaluate --file mobilenetv3_2025-05-20_09-38-24_val_0.3869.pth
python runner.py --models cnn --mode evaluate --file cnn_2025-05-20_09-41-03_val_0.3092.pth