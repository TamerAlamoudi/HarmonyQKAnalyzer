#!/usr/bin/env python3
# custom_model_example.py
"""
Example script demonstrating how to load custom model weights and analyze them.
"""
import os
import argparse
import json

from config import BASE_DIR, DATA_PATH, FIGURE_PATH, IMAGENET_VAL_PATH, get_device
from compute_svd import compute_and_save_svd
from compute_optimal_images import compute_activations, visualize_optimal_images
from logging_utils import setup_logger, get_timestamp

def main():
    """Main function to demonstrate loading custom model weights."""
    parser = argparse.ArgumentParser(description="Process a custom model with HarmonyQKAnalyzer")
    parser.add_argument("--model_type", type=str, default="vit", choices=["vit", "deit", "clip"],
                        help="Type of model architecture")
    parser.add_argument("--weights_path", type=str, required=True,
                        help="Path to the custom weights file (.pth, .pt, or .bin)")
    parser.add_argument("--config_file", type=str,
                        help="Path to custom model config JSON file (optional)")
    parser.add_argument("--model_name", type=str, default="custom_model",
                        help="Name to use for this model in files and logs")
    parser.add_argument("--data_path", type=str, default=DATA_PATH,
                        help="Path to save the results")
    parser.add_argument("--figure_path", type=str, default=FIGURE_PATH,
                        help="Path to save the output figures")
    parser.add_argument("--dataset_path", type=str, default=IMAGENET_VAL_PATH,
                        help="Path to the dataset")
    parser.add_argument("--log_dir", type=str, default=os.path.join(BASE_DIR, "logs"),
                        help="Directory for log files")
    parser.add_argument("--skip_svd", action="store_true",
                        help="Skip SVD computation (use if already computed)")
    
    # Arguments for visualization
    parser.add_argument("--layer", type=int, help="Layer to visualize (if None, will process all layers)")
    parser.add_argument("--head", type=int, help="Head to visualize (if None, will process all heads)")
    parser.add_argument("--sign", type=str, choices=["p", "n"], help="Sign to visualize (if None, will process both)")
    parser.add_argument("--modes", type=int, nargs="+", help="List of modes to visualize (if None, will use top 20)")
    args = parser.parse_args()
    
    # Set up logger
    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = get_timestamp()
    log_file = os.path.join(args.log_dir, f"custom_model_{timestamp}.log")
    logger = setup_logger("custom_model", log_file)
    
    # Determine device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Load model config if provided
    model_config = None
    if args.config_file:
        try:
            with open(args.config_file, 'r') as f:
                model_config = json.load(f)
                logger.info(f"Loaded config from {args.config_file}")
        except Exception as e:
            logger.error(f"Error loading config file {args.config_file}: {e}")
            logger.info("Using default model config instead")
    
    # Generate the full model name based on the model type
    full_model_name = f"{args.model_type}/{args.model_name}"
    logger.info(f"Using model name: {full_model_name}")
    
    # Step 1: Compute SVD if not skipped
    if not args.skip_svd:
        logger.info("=== Step 1: Computing SVD ===")
        compute_and_save_svd(
            [full_model_name],
            args.data_path,
            device,
            [args.weights_path],
            [model_config] if model_config else None,
            log_file
        )
    
    # Step 2: Compute activations
    logger.info("=== Step 2: Computing activations ===")
    try:
        compute_activations(
            full_model_name,
            args.dataset_path,
            os.path.join(args.data_path, "optimal_images"),
            args.weights_path,
            model_config,
            device=device,
            log_file=log_file
        )
    except Exception as e:
        logger.error(f"Error computing activations: {e}")
    
    # Step 3: Generate visualizations
    logger.info("=== Step 3: Generating visualizations ===")
    try:
        visualize_optimal_images(
            full_model_name,
            args.dataset_path,
            args.figure_path,
            args.data_path,
            args.layer,
            args.head,
            args.sign,
            args.modes,
            args.weights_path,
            model_config,
            device=device,
            log_file=log_file
        )
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
    
    logger.info("Custom model processing completed")

if __name__ == "__main__":
    main() 