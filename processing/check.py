import torch

# This file checks if GPU can be used to run the HF model
print("CUDA available:", torch.cuda.is_available())
print("Torch version:", torch.__version__)
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
