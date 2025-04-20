# Setup pytorch properly
# Env uses 2.1.0
pip uninstall torch torchvision torchaudio torchtext -y

# Install 2.6.0 version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Go up two levels to install the other requirements
cd ../..
pip install -r requirements.txt

# Download the processed dataset
cd src/
python data_setup.py