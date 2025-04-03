# models/model_loader.py
"""
Module for loading and initializing various vision transformer models.
"""
from typing import Callable, Tuple

import torch
from transformers import (
    AutoImageProcessor,
    CLIPModel,
    CLIPProcessor,
    DeiTForImageClassificationWithTeacher,
    Dinov2Model,
    ViTForImageClassification,
)


def load_model(
    model_name: str, device: torch.device = torch.device("cpu")
) -> Tuple[torch.nn.Module, Callable]:
    """
    Load a model and its processor based on the model name.

    Args:
        model_name: Name of the model to load
        device: Device to load the model on

    Returns:
        Tuple containing:
            - The loaded model
            - The processor function for preparing inputs
    """
    # Load based on model type
    if "clip" in model_name:
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
                padding=True,
            )

    elif "dinov2" in model_name:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = Dinov2Model.from_pretrained(model_name)
        model = model.to(device)

    elif "dino" in model_name:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = ViTForImageClassification.from_pretrained(model_name)
        model = model.to(device)

    elif "vit" in model_name:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = ViTForImageClassification.from_pretrained(model_name)
        model = model.to(device)

    elif "deit" in model_name:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = DeiTForImageClassificationWithTeacher.from_pretrained(model_name)
        model = model.to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    return model, processor


def get_start_index(model_name: str) -> int:
    """
    Get the token start index for a model.

    Args:
        model_name: Name of the model

    Returns:
        Start index for tokens (usually 1 for most models, 2 for deit)
    """
    if "deit" in model_name:
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
    from models.model_analysis import (
        LayerNormHook,
        LayerNormHook_clip,
        LayerNormHook_deit,
    )

    if "deit" in model_name:
        return LayerNormHook_deit(model)
    elif "clip" in model_name:
        return LayerNormHook_clip(model)
    else:
        return LayerNormHook(model)
