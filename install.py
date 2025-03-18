import os
import subprocess


def install_pytorch():
    try:
        # Check if GPU is available
        gpu_available = subprocess.run(
            ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if gpu_available.returncode == 0:
            print("GPU detected, installing PyTorch with CUDA support...")
            subprocess.run(["pip", "install", "torch", "torchvision", "torchaudio",
                           "--index-url", "https://download.pytorch.org/whl/cu118"])
        else:
            print("No GPU detected, installing CPU version of PyTorch...")
            subprocess.run(["pip", "install", "torch",
                           "torchvision", "torchaudio"])
    except Exception as e:
        print(f"Error detecting GPU: {e}")
        print("Installing CPU version of PyTorch...")
        subprocess.run(["pip", "install", "torch",
                       "torchvision", "torchaudio"])


def install_dependencies():
    # Get pytorch
    install_pytorch()

    # Install other dependencies
    print("Installing other dependencies")
    subprocess.run(["pip", "install", "-r", "requirements.txt"])


if __name__ == "__main__":
    install_dependencies()
