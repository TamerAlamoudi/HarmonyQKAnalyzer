# config.py
"""
Configuration settings for the HarmonyQKAnalyzer project.
"""
import os
import torch

# Paths
BASE_DIR = "/opt/scratch/labs/oxs/vit_2023/tamer/HarmonyQKAnalyzer"
BASE_DIR_VAL= "/opt/labs/oxs/vit_2023"
DATA_PATH = os.path.join(BASE_DIR, "results/DissectingVitWithImagenette/data_test")
FIGURE_PATH = os.path.join(BASE_DIR, "results/DissectingVitWithImagenette/figure_test")
LOGS_PATH = os.path.join(BASE_DIR, "results/DissectingVitWithImagenette/logs")
BASE_DIR_VAL = "/opt/labs/oxs/vit_2023"
# IMAGENET_VAL_PATH = os.path.join(BASE_DIR_VAL, "visualization")
# IMAGENET_LIST_PATH = os.path.join(IMAGENET_VAL_PATH, "imagenet_list_val.pkl")
# Update the actual path to the ImageNet validation images
IMAGENET_VAL_PATH = "/opt/labs/oxs/vit_2023/ImageNet1k/ILSVRC/Data/CLS-LOC/val"
# Keep the path to the pickle file as is
IMAGENET_LIST_PATH = "/opt/labs/oxs/vit_2023/visualization/imagenet_list_val.pkl"

# Device configuration
DEVICE = 'cuda:0'  # Change to 'cuda' or 'mps' if available
USE_GPU = True  # Set to True to use GPU if available

# Default model configuration
DEFAULT_MODELS = [
    "google/vit-base-patch16-224",
    # "google/vit-base-patch32-384",
    # "google/vit-large-patch16-224",
    # "google/vit-large-patch32-384",
    # "google/vit-huge-patch14-224-in21k",
    # "facebook/dino-vitb16",
    # "facebook/dino-vits16",
    # "openai/clip-vit-base-patch16",
    # "openai/clip-vit-base-patch32",
    # "openai/clip-vit-large-patch14",
    # "facebook/deit-base-distilled-patch16-224",
    # "facebook/deit-small-distilled-patch16-224",
    # "facebook/deit-tiny-distilled-patch16-224"
]

# Create required directories
for path in [DATA_PATH, FIGURE_PATH, LOGS_PATH]:
    os.makedirs(path, exist_ok=True)
    
# Function to get device based on availability
def get_device() -> torch.device:
    """Get the best available device."""
    if USE_GPU:
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
    return torch.device('cpu')

# Initialize device
DEVICE = get_device()