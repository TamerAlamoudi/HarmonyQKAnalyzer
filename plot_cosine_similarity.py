#!/usr/bin/env python3
# plot_cosine_similarity.py
"""
Script to generate plots of cosine similarity between U and V matrices.
"""
import os
import argparse
from typing import List, Optional

from config import BASE_DIR, DATA_PATH, FIGURE_PATH, DEFAULT_MODELS
from visualization import plot_cosine_similarity, plot_spectrum_and_cosine
from logging_utils import setup_logger, get_timestamp

def generate_cosine_plots(
    model_names: List[str],
    data_path: str,
    figure_path: str,
    generate_spectrum: bool = True,
    weights_paths: Optional[List[str]] = None,
    file_prefixes: Optional[List[str]] = None,
    log_file: Optional[str] = None
) -> None:
    """
    Generate cosine similarity plots for the specified models.
    
    Args:
        model_names: List of model names to process
        data_path: Path to the data directory containing UVS files
        figure_path: Path to save the output figures
        generate_spectrum: Whether to generate spectrum plots for each model
        weights_paths: List of paths to custom weights files (optional)
        file_prefixes: List of file prefixes to use for consistency (optional)
        log_file: Path to log file
    """
    # Set up logger
    logger = setup_logger("plot_cosine", log_file)
    logger.info(f"Starting cosine similarity plotting for {len(model_names)} models")
    
    # Initialize weights_paths and file_prefixes if not provided
    if weights_paths is None:
        weights_paths = [None] * len(model_names)
    
    if file_prefixes is None:
        file_prefixes = [None] * len(model_names)
    
    # Make sure the lists have the same length
    if len(weights_paths) != len(model_names):
        logger.warning(f"Number of weights paths ({len(weights_paths)}) does not match number of models ({len(model_names)})")
        # Pad with None if too short
        if len(weights_paths) < len(model_names):
            weights_paths.extend([None] * (len(model_names) - len(weights_paths)))
        # Truncate if too long
        weights_paths = weights_paths[:len(model_names)]
    
    if len(file_prefixes) != len(model_names):
        logger.warning(f"Number of file prefixes ({len(file_prefixes)}) does not match number of models ({len(model_names)})")
        # Pad with None if too short
        if len(file_prefixes) < len(model_names):
            file_prefixes.extend([None] * (len(model_names) - len(file_prefixes)))
        # Truncate if too long
        file_prefixes = file_prefixes[:len(model_names)]
    
    # Generate averaged cosine similarity plot
    logger.info("Generating average cosine similarity plot")
    try:
        output_path = plot_cosine_similarity(
            model_names, 
            data_path, 
            figure_path, 
            weighted=False,
            weights_paths=weights_paths,
            file_prefixes=file_prefixes,
            logger=logger
        )
        logger.info(f"Average cosine similarity plot saved to {output_path}")
    except Exception as e:
        logger.error(f"Error generating average cosine similarity plot: {e}")
    
    # Generate weighted cosine similarity plot
    logger.info("Generating weighted average cosine similarity plot")
    try:
        output_path = plot_cosine_similarity(
            model_names, 
            data_path, 
            figure_path, 
            weighted=True,
            weights_paths=weights_paths,
            file_prefixes=file_prefixes,
            logger=logger
        )
        logger.info(f"Weighted average cosine similarity plot saved to {output_path}")
    except Exception as e:
        logger.error(f"Error generating weighted average cosine similarity plot: {e}")
    
    # Generate spectrum plots for each model if requested
    if generate_spectrum:
        for i, model_name in enumerate(model_names):
            logger.info(f"Generating spectrum plot for {model_name}")
            try:
                output_path = plot_spectrum_and_cosine(
                    model_name, 
                    data_path, 
                    figure_path,
                    weights_path=weights_paths[i],
                    file_prefix=file_prefixes[i],
                    logger=logger
                )
                logger.info(f"Spectrum plot for {model_name} saved to {output_path}")
            except Exception as e:
                logger.error(f"Error generating spectrum plot for {model_name}: {e}")
    
    logger.info("Cosine similarity plotting completed")

def main():
    """Main function to parse arguments and run cosine similarity plotting."""
    parser = argparse.ArgumentParser(description="Plot cosine similarity for transformer models")
    parser.add_argument("--models", nargs="+", help="List of model names to process")
    parser.add_argument("--weights_paths", nargs="+", help="List of paths to custom weights files")
    parser.add_argument("--file_prefixes", nargs="+", help="List of file prefixes to use for consistency")
    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="Path to the data directory containing UVS files")
    parser.add_argument("--figure_path", type=str, default=FIGURE_PATH, help="Path to save the output figures")
    parser.add_argument("--no_spectrum", action="store_true", help="Do not generate spectrum plots")
    parser.add_argument("--log_dir", type=str, default=os.path.join(BASE_DIR, "logs"), help="Directory for log files")
    args = parser.parse_args()
    
    # Use provided models or default ones
    model_names = args.models if args.models else DEFAULT_MODELS
    
    # Create log directory if it doesn't exist
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Generate log file path
    timestamp = get_timestamp()
    log_file = os.path.join(args.log_dir, f"plot_cosine_{timestamp}.log")
    
    # Run cosine similarity plotting
    generate_cosine_plots(
        model_names, 
        args.data_path, 
        args.figure_path, 
        not args.no_spectrum,
        args.weights_paths,
        args.file_prefixes,
        log_file
    )

    # Example usage for comparing canonical and harmonized models:
    """
    # To compare canonical and harmonized ViT-B/16 models:
    # python plot_cosine_similarity.py --models "vit-base-patch16-224" "vit-base-patch16-224" --file_prefixes "vit_b16_canonical" "vit_b16_harmonized"
    
    # You can also run this directly from Python:
    # generate_cosine_plots(
    #     ["vit-base-patch16-224", "vit-base-patch16-224"],
    #     DATA_PATH,
    #     FIGURE_PATH,
    #     generate_spectrum=True,
    #     weights_paths=[None, None],
    #     file_prefixes=["vit_b16_canonical", "vit_b16_harmonized"]
    # )
    """

if __name__ == "__main__":
    main() 