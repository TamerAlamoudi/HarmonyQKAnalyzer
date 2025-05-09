# models/model_loader.py
"""
Module for loading and initializing various vision transformer models.
"""
import os
import torch
from typing import Tuple, Callable, Optional, Dict
from transformers import (
    AutoImageProcessor, 
    ViTForImageClassification, 
    CLIPProcessor, 
    CLIPModel, 
    DeiTForImageClassificationWithTeacher, 
    Dinov2Model,
    ViTConfig,
    CLIPVisionConfig,
    DeiTConfig,
)

def load_model(
    model_name: str, 
    device: torch.device = torch.device('cpu'),
    weights_path: Optional[str] = None,
    model_config: Optional[Dict] = None
) -> Tuple[torch.nn.Module, Callable]:
    """
    Load a model and its processor based on the model name.
    
    Args:
        model_name: Name of the model to load
        device: Device to load the model on
        weights_path: Path to custom weights file (optional)
        model_config: Dictionary with model configuration (optional)
        
    Returns:
        Tuple containing:
            - The loaded model
            - The processor function for preparing inputs
    """
    # Determine if we're loading a custom model or a pretrained one
    is_custom_model = weights_path is not None
    
    # For custom models, we need to determine their processor
    if is_custom_model:
        return load_custom_model(model_name, weights_path, model_config, device)
    
    # For pretrained models, load from Huggingface
    # Load based on model type
    if 'clip' in model_name:
        model = CLIPModel.from_pretrained(model_name)
        processor_clip = CLIPProcessor.from_pretrained(model_name)
        model = model.to(device)
        model = model.vision_model
        
        # Define processor function for clip models
        def processor(images, return_tensors="pt"):
            return processor_clip(
                text=["a photo of a cat", "a photo of a dog"], 
                images=images, 
                return_tensors=return_tensors, 
                padding=True
            )
            
    elif 'dinov2' in model_name:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = Dinov2Model.from_pretrained(model_name)
        model = model.to(device)
        
    elif 'dino' in model_name:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = ViTForImageClassification.from_pretrained(model_name)
        model = model.to(device)
        
    elif 'vit' in model_name:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = ViTForImageClassification.from_pretrained(model_name)
        model = model.to(device)
        
    elif 'deit' in model_name:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = DeiTForImageClassificationWithTeacher.from_pretrained(model_name)
        model = model.to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    return model, processor

def load_custom_model(
    model_name: str,
    weights_path: str,
    model_config: Optional[Dict] = None,
    device: torch.device = torch.device('cpu')
) -> Tuple[torch.nn.Module, Callable]:
    """
    Load a custom model with weights from a local file.
    
    Args:
        model_name: Base model architecture name (vit, clip, deit, etc.)
        weights_path: Path to the weights file
        model_config: Dictionary with model configuration
        device: Device to load the model on
        
    Returns:
        Tuple containing:
            - The loaded model
            - The processor function for preparing inputs
    """
    # Verify that weights file exists
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    # Create the model based on the type
    if 'clip' in model_name:
        # For CLIP models, we need a config
        if model_config is None:
            raise ValueError("Config is required for custom CLIP models")
        
        config = CLIPVisionConfig(**model_config)
        model = CLIPModel(config)
        model = model.vision_model
        
        # Create a processor for the CLIP model
        # We'll use a pretrained processor since custom processors are complex
        reference_model = "openai/clip-vit-base-patch16"
        processor_clip = CLIPProcessor.from_pretrained(reference_model)
        
        def processor(images, return_tensors="pt"):
            return processor_clip(
                text=["a photo of a cat", "a photo of a dog"], 
                images=images, 
                return_tensors=return_tensors, 
                padding=True
            )
    
    elif 'vit' in model_name:
        # Create ViT model
        if model_config is None:
            # Default ViT-base config
            config = ViTConfig(
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                num_labels=1000
            )
        else:
            config = ViTConfig(**model_config)
        
        model = ViTForImageClassification(config)
        
        # Use a pretrained processor for simplicity
        reference_model = "google/vit-base-patch16-224"
        processor = AutoImageProcessor.from_pretrained(reference_model)
    
    elif 'deit' in model_name:
        # Create DeiT model
        if model_config is None:
            # Default DeiT-base config
            config = DeiTConfig(
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                num_labels=1000
            )
        else:
            config = DeiTConfig(**model_config)
        
        model = DeiTForImageClassificationWithTeacher(config)
        
        # Use a pretrained processor for simplicity
        reference_model = "facebook/deit-base-distilled-patch16-224"
        processor = AutoImageProcessor.from_pretrained(reference_model)
    
    else:
        raise ValueError(f"Unsupported custom model type: {model_name}")
    
    # Load weights
    try:
        # Try to load state dict
        state_dict = torch.load(weights_path, map_location=device)
        
        # Handle the case where the state_dict might be nested
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Some custom weights might have different key names, attempt to load them
        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            print(f"Warning: Could not load weights with strict=True. Error: {e}")
            print("Attempting to load with strict=False")
            model.load_state_dict(state_dict, strict=False)
            
    except Exception as e:
        raise RuntimeError(f"Failed to load weights from {weights_path}: {e}")
    
    # Move model to specified device
    model = model.to(device)
    
    return model, processor

def get_model_name_for_files(model_name: str, weights_path: Optional[str] = None, custom_file_prefix: Optional[str] = None) -> str:
    """
    Generate a safe model name for file paths, ensuring uniqueness for custom models.
    
    Args:
        model_name: Original model name
        weights_path: Path to weights file (if custom model)
        custom_file_prefix: Optional custom prefix to use instead of generated one
        
    Returns:
        Safe model name for file paths
    """
    # If a custom file prefix is provided, use it directly
    if custom_file_prefix:
        return custom_file_prefix
        
    # Replace slashes with underscores
    safe_name = model_name.replace("/", "_")
    
    # For custom models, add a suffix based on the weights filename
    if weights_path:
        weights_filename = os.path.basename(weights_path)
        weights_name = os.path.splitext(weights_filename)[0]
        safe_name = f"{safe_name}_custom_{weights_name}"
    
    return safe_name

def get_start_index(model_name: str) -> int:
    """
    Get the token start index for a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Start index for tokens (usually 1 for most models, 2 for deit)
    """
    if 'deit' in model_name:
        return 2
    return 1

def create_hook_model(model: torch.nn.Module, model_name: str) -> torch.nn.Module:
    """
    Create a hook model for the given model type.
    
    Args:
        model: Base model
        model_name: Name of the model
        
    Returns:
        Hook model appropriate for the model type
    """
    from model_analysis import LayerNormHook, LayerNormHook_deit, LayerNormHook_clip
    
    if 'deit' in model_name:
        return LayerNormHook_deit(model)
    elif 'clip' in model_name:
        return LayerNormHook_clip(model)
    else:
        return LayerNormHook(model)