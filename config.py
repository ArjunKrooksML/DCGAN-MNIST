import torch
import os

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
LEARNING_RATE = 2e-4  
BATCH_SIZE = 128
IMAGE_SIZE = 28       
CHANNELS = 1      
Z_DIM = 100           
EPOCHS = 20       
DISC_FEATURES = 64   
GEN_FEATURES = 64     

# Directories
OUTPUT_DIR = "Output"
DATASET_DIR = "dataset/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Using device: {DEVICE}")
print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
print(f"Dataset directory: {os.path.abspath(DATASET_DIR)}")