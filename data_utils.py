# utils/data_utils.py
"""
Utility functions for data loading and manipulation.
"""
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Any, Callable

def load_pickle(file_path: str) -> Any:
    """
    Load data from a pickle file.
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        Loaded data
    """
    with open(file_path, 'rb') as fp:
        data = pickle.load(fp)
    return data

def save_pickle(data: Any, file_path: str) -> None:
    """
    Save data to a pickle file.
    
    Args:
        data: Data to save
        file_path: Path to save the pickle file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'wb') as fp:
        pickle.dump(data, fp)

class ImageDataset(Dataset):
    """Dataset for loading and processing images."""
    
    def __init__(self, image_list: List[str], processor: Callable, image_dir: str):
        """
        Initialize the dataset.
        
        Args:
            image_list: List of image filenames
            processor: Function to process images
            image_dir: Directory containing the images
        """
        self.image_list = image_list
        self.processor = processor
        self.image_dir = image_dir

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_list)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Load and process an image.
        
        Args:
            idx: Index of the image to load
            
        Returns:
            Processed image tensor
        """
        im_file = self.image_list[idx]
        image_path = os.path.join(self.image_dir, im_file)
        
        # Load image
        try:
            im = plt.imread(image_path)
        except Exception as e:
            im = np.array(Image.open(image_path).convert('RGB'))
            
        # Handle grayscale images
        if len(im.shape) < 3:  # grayscale images
            im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
            
        # Process image
        processed_image = self.processor(images=im, return_tensors="pt")["pixel_values"].squeeze(0)
        return processed_image

def save_numpy_array(array: np.ndarray, file_path: str) -> None:
    """
    Save numpy array to file with proper directory creation.
    
    Args:
        array: NumPy array to save
        file_path: Path to save the array
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Save array
    np.save(file_path, array)

def load_numpy_array(file_path: str) -> np.ndarray:
    """
    Load numpy array from file.
    
    Args:
        file_path: Path to the array file
        
    Returns:
        Loaded NumPy array
    """
    return np.load(file_path)