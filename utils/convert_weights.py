#!/usr/bin/env python3
# convert_weights.py
"""
Utility scripts for converting model weights to be compatible 
with the HarmonyQKAnalyzer framework.
"""
import os
import argparse
import json
import torch
from transformers import ViTConfig, ViTForImageClassification
from typing import Dict, Any, Optional

def convert_pytorch_weights(
    input_path: str,
    output_path: str,
    model_type: str = "vit",
    config_path: Optional[str] = None,
    save_config: bool = True
) -> None:
    """
    Convert PyTorch saved weights to a format compatible with HarmonyQKAnalyzer.
    
    Args:
        input_path: Path to input weights file (.pth, .pt, or .bin)
        output_path: Path to save converted weights
        model_type: Type of model architecture (vit, deit, clip)
        config_path: Path to model config JSON file (optional)
        save_config: Whether to save a separate config file
    """
    print(f"Loading weights from {input_path}")
    state_dict = torch.load(input_path, map_location=torch.device("cpu"))
    
    # Handle different state_dict formats
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Determine configuration
    config = None
    if config_path:
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Try to infer config from the state dict
        config = infer_config_from_state_dict(state_dict, model_type)
    
    # Save configuration if requested
    if save_config and config:
        config_output_path = f"{os.path.splitext(output_path)[0]}_config.json"
        with open(config_output_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Saved model configuration to {config_output_path}")
    
    # Save weights
    torch.save(state_dict, output_path)
    print(f"Saved converted weights to {output_path}")

def convert_from_huggingface(
    model_name: str,
    output_path: str,
    save_config: bool = True
) -> None:
    """
    Download a model from Hugging Face and save its weights in a format
    compatible with HarmonyQKAnalyzer.
    
    Args:
        model_name: Name of the Hugging Face model
        output_path: Path to save converted weights
        save_config: Whether to save a separate config file
    """
    print(f"Downloading model {model_name} from Hugging Face")
    try:
        from transformers import AutoModel, AutoConfig
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Load model and config
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Save the model weights
        torch.save(model.state_dict(), output_path)
        print(f"Saved model weights to {output_path}")
        
        # Save configuration if requested
        if save_config:
            config_dict = config.to_dict()
            config_output_path = f"{os.path.splitext(output_path)[0]}_config.json"
            with open(config_output_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            print(f"Saved model configuration to {config_output_path}")
    
    except Exception as e:
        print(f"Error downloading model: {e}")

def infer_config_from_state_dict(state_dict: Dict[str, Any], model_type: str) -> Dict[str, Any]:
    """
    Attempt to infer model configuration from the state dictionary.
    
    Args:
        state_dict: Model state dictionary
        model_type: Type of model architecture (vit, deit, clip)
        
    Returns:
        Inferred model configuration
    """
    config = {}
    
    # Attempt to infer hidden size
    for key in state_dict.keys():
        if 'embeddings.position_embeddings' in key and len(state_dict[key].shape) > 0:
            seq_len = state_dict[key].shape[1]
            if model_type == "vit":
                # ViT models typically have a cls token, so patch count is seq_len - 1
                patch_count = seq_len - 1
                # Assume square images
                patch_size = 16  # Default for ViT-Base
                image_size = int((patch_count)**0.5 * patch_size)
                config["image_size"] = image_size
                config["patch_size"] = patch_size
            break
    
    # Infer hidden dimension
    for key in state_dict.keys():
        if 'encoder.layer.0.attention.self.query.weight' in key:
            hidden_size = state_dict[key].shape[0]
            config["hidden_size"] = hidden_size
            break
    
    # Infer number of attention heads
    for key in state_dict.keys():
        if 'encoder.layer.0.attention.self.query.weight' in key:
            # Assuming head size is typically 64
            head_size = 64
            num_heads = config.get("hidden_size", 768) // head_size
            config["num_attention_heads"] = num_heads
            break
    
    # Infer number of layers
    max_layer = 0
    for key in state_dict.keys():
        if 'encoder.layer.' in key:
            layer_num = int(key.split('encoder.layer.')[1].split('.')[0])
            max_layer = max(max_layer, layer_num)
    
    config["num_hidden_layers"] = max_layer + 1
    
    # Set defaults for missing values
    if "hidden_size" not in config:
        config["hidden_size"] = 768  # Default for ViT-Base
    
    if "num_attention_heads" not in config:
        config["num_attention_heads"] = 12  # Default for ViT-Base
    
    if "num_hidden_layers" not in config:
        config["num_hidden_layers"] = 12  # Default for ViT-Base
    
    if "image_size" not in config:
        config["image_size"] = 224  # Default for ViT-Base
    
    if "patch_size" not in config:
        config["patch_size"] = 16  # Default for ViT-Base
    
    # Add other common configuration parameters
    config["hidden_act"] = "gelu"
    config["layer_norm_eps"] = 1e-12
    config["initializer_range"] = 0.02
    
    print("Inferred configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    return config

def main():
    """Main function to parse arguments and run the conversion."""
    parser = argparse.ArgumentParser(description="Convert model weights for HarmonyQKAnalyzer")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to input weights file (.pth, .pt, or .bin) or Hugging Face model name")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save converted weights")
    parser.add_argument("--model_type", type=str, default="vit", choices=["vit", "deit", "clip"],
                        help="Type of model architecture")
    parser.add_argument("--config_path", type=str,
                        help="Path to model config JSON file (optional)")
    parser.add_argument("--from_huggingface", action="store_true",
                        help="Whether the input is a Hugging Face model name")
    parser.add_argument("--no_save_config", action="store_true",
                        help="Don't save a separate config file")
    
    args = parser.parse_args()
    
    if args.from_huggingface:
        convert_from_huggingface(
            args.input_path,
            args.output_path,
            not args.no_save_config
        )
    else:
        convert_pytorch_weights(
            args.input_path,
            args.output_path,
            args.model_type,
            args.config_path,
            not args.no_save_config
        )

if __name__ == "__main__":
    main() 