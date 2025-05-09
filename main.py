#!/usr/bin/env python3
# main.py
"""
Main script to run the HarmonyQKAnalyzer pipeline.
"""
import os
import argparse
import json
import logging
from typing import List, Optional, Dict, Any

from config import BASE_DIR, DATA_PATH, FIGURE_PATH, IMAGENET_VAL_PATH, DEFAULT_MODELS, get_device
from compute_svd import compute_and_save_svd
from compute_optimal_images import compute_activations, visualize_optimal_images
from plot_cosine_similarity import plot_cosine_similarity
from visualization import plot_spectrum_and_cosine
from logging_utils import setup_logger, get_timestamp

def run_pipeline(
    model_names: List[str],
    data_path: str,
    figure_path: str,
    dataset_path: str,
    skip_svd: bool = False,
    skip_cosine: bool = False,
    skip_activations: bool = False,
    skip_visualizations: bool = False,
    layer: Optional[int] = None,
    head: Optional[int] = None,
    sign: Optional[str] = None,
    modes: Optional[List[int]] = None,
    weights_paths: Optional[List[str]] = None,
    model_configs: Optional[List[Dict[str, Any]]] = None,
    file_prefixes: Optional[List[str]] = None,
    device: str = "cuda",
    log_file: Optional[str] = None
) -> None:
    """
    Run the complete HarmonyQKAnalyzer pipeline.
    
    Args:
        model_names: List of model names to process
        data_path: Path to the data directory
        figure_path: Path to save the output figures
        dataset_path: Path to the dataset
        skip_svd: Whether to skip SVD computation
        skip_cosine: Whether to skip cosine similarity plots
        skip_activations: Whether to skip activation computation
        skip_visualizations: Whether to skip visualizations
        layer: Layer index for visualizations
        head: Head index for visualizations
        sign: Sign of the activations ('p' or 'n')
        modes: List of mode indices to visualize
        weights_paths: List of paths to custom model weights
        model_configs: List of custom model configurations
        file_prefixes: List of file prefixes to use for consistency
        device: Device to use for computation
        log_file: Path to log file
    """
    logger = logging.getLogger("main")
    
    # Ensure weights_paths and model_configs match model_names in length if provided
    if weights_paths:
        if len(weights_paths) != len(model_names):
            logger.warning(f"Number of weights_paths ({len(weights_paths)}) doesn't match model_names ({len(model_names)})")
            # Adjust to the shorter length
            min_len = min(len(weights_paths), len(model_names))
            weights_paths = weights_paths[:min_len]
            model_names = model_names[:min_len]
    else:
        weights_paths = [None] * len(model_names)
    
    if model_configs:
        if len(model_configs) != len(model_names):
            logger.warning(f"Number of model_configs ({len(model_configs)}) doesn't match model_names ({len(model_names)})")
            # Adjust to the shorter length
            min_len = min(len(model_configs), len(model_names))
            model_configs = model_configs[:min_len]
            model_names = model_names[:min_len]
    else:
        model_configs = [None] * len(model_names)
    
    if file_prefixes:
        if len(file_prefixes) != len(model_names):
            logger.warning(f"Number of file_prefixes ({len(file_prefixes)}) doesn't match model_names ({len(model_names)})")
            # Adjust to the shorter length
            min_len = min(len(file_prefixes), len(model_names))
            file_prefixes = file_prefixes[:min_len]
    else:
        file_prefixes = [None] * len(model_names)
    
    # Step 1: Compute SVD matrices
    if not skip_svd:
        logger.info("Computing SVD matrices...")
        compute_and_save_svd(model_names, data_path, device, weights_paths, model_configs, file_prefixes, log_file)
    
    # Step 2: Generate cosine similarity plots
    if not skip_cosine:
        logger.info("Generating cosine similarity plots...")
        plot_cosine_similarity(model_names, data_path, figure_path, device, weights_paths, file_prefixes, log_file)
        plot_spectrum_and_cosine(model_names, data_path, figure_path, weights_paths, file_prefixes, log_file)
    
    # Steps 3 and 4: Compute activations and generate visualizations for each model
    for i, model_name in enumerate(model_names):
        weights_path = weights_paths[i]
        model_config = model_configs[i] if model_configs else None
        file_prefix = file_prefixes[i] if file_prefixes else None
        
        # Step 3: Compute activations
        if not skip_activations:
            logger.info(f"Computing activations for model {model_name}...")
            try:
                compute_activations(
                    model_name,
                    dataset_path,
                    os.path.join(data_path, "optimal_images"),
                    weights_path,
                    model_config,
                    device=device,
                    log_file=log_file,
                    file_prefix=file_prefix
                )
            except Exception as e:
                logger.error(f"Error computing activations for model {model_name}: {e}")
        
        # Step 4: Generate visualizations
        if not skip_visualizations:
            logger.info(f"Generating visualizations for model {model_name}...")
            try:
                visualize_optimal_images(
                    model_name,
                    dataset_path,
                    figure_path,
                    data_path,
                    layer,
                    head,
                    sign,
                    modes,
                    weights_path,
                    model_config,
                    device=device,
                    log_file=log_file,
                    file_prefix=file_prefix
                )
            except Exception as e:
                logger.error(f"Error generating visualizations for model {model_name}: {e}")

def main():
    """Main function to parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(description="Run the HarmonyQKAnalyzer pipeline")
    
    # Model selection arguments
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--models", type=str, nargs="+", help="List of model names to process")
    model_group.add_argument("--model_name", type=str, help="Single model name to process")
    
    # Custom model arguments
    parser.add_argument("--weights_paths", type=str, nargs="+", help="List of paths to custom model weights")
    parser.add_argument("--weights_path", type=str, help="Path to custom model weights for a single model")
    parser.add_argument("--config_files", type=str, nargs="+", help="List of paths to custom model config files")
    parser.add_argument("--config_file", type=str, help="Path to custom model config file for a single model")
    parser.add_argument("--file_prefixes", type=str, nargs="+", help="List of file prefixes to use for consistency")
    parser.add_argument("--file_prefix", type=str, help="File prefix to use for consistency with a single model")
    
    # Path arguments
    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="Path to the data directory")
    parser.add_argument("--figure_path", type=str, default=FIGURE_PATH, help="Path to save the output figures")
    parser.add_argument("--dataset_path", type=str, default=IMAGENET_VAL_PATH, help="Path to the dataset")
    parser.add_argument("--log_dir", type=str, default=os.path.join(BASE_DIR, "logs"),
                        help="Directory for log files")
    
    # Pipeline control arguments
    parser.add_argument("--skip_svd", "--no_svd", action="store_true", help="Skip SVD computation")
    parser.add_argument("--skip_cosine", "--no_cosine", action="store_true", help="Skip cosine similarity plots")
    parser.add_argument("--skip_activations", "--no_activations", action="store_true", 
                        help="Skip activation computation")
    parser.add_argument("--skip_visualizations", "--no_visualizations", action="store_true", 
                        help="Skip visualizations")
    
    # Visualization arguments
    parser.add_argument("--layer", type=int, help="Layer index for visualizations")
    parser.add_argument("--head", type=int, help="Head index for visualizations")
    parser.add_argument("--sign", type=str, choices=["p", "n"], help="Sign of the activations ('p' or 'n')")
    parser.add_argument("--modes", type=int, nargs="+", help="List of mode indices to visualize")
    
    args = parser.parse_args()
    
    # Set up logger
    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = get_timestamp()
    log_file = os.path.join(args.log_dir, f"main_{timestamp}.log")
    logger = setup_logger("main", log_file)
    
    # Prepare model names
    if args.model_name:
        model_names = [args.model_name]
    elif args.models:
        model_names = args.models
    else:
        model_names = DEFAULT_MODELS
        logger.info(f"Using default models: {model_names}")
    
    # Prepare weights paths
    weights_paths = None
    if args.weights_path:
        weights_paths = [args.weights_path]
    elif args.weights_paths:
        weights_paths = args.weights_paths
    
    # Prepare file prefixes
    file_prefixes = None
    if args.file_prefix:
        file_prefixes = [args.file_prefix]
    elif args.file_prefixes:
        file_prefixes = args.file_prefixes
    
    # Prepare model configs
    model_configs = None
    if args.config_file or args.config_files:
        model_configs = []
        config_files = [args.config_file] if args.config_file else args.config_files
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    model_configs.append(json.load(f))
                    logger.info(f"Loaded config from {config_file}")
            except Exception as e:
                logger.error(f"Error loading config file {config_file}: {e}")
                model_configs.append(None)
    
    # Determine device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Run the pipeline
    run_pipeline(
        model_names=model_names,
        data_path=args.data_path,
        figure_path=args.figure_path,
        dataset_path=args.dataset_path,
        skip_svd=args.skip_svd,
        skip_cosine=args.skip_cosine,
        skip_activations=args.skip_activations,
        skip_visualizations=args.skip_visualizations,
        layer=args.layer,
        head=args.head,
        sign=args.sign,
        modes=args.modes,
        weights_paths=weights_paths,
        model_configs=model_configs,
        file_prefixes=file_prefixes,
        device=device,
        log_file=log_file
    )
    
    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    main() 