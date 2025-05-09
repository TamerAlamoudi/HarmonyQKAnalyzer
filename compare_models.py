#!/usr/bin/env python3
# compare_models.py
"""
Script to compare canonical and harmonized ViT models.
This generates all plots needed for the paper including:
1. Cosine similarity (weighted and unweighted)
2. KL divergence
3. Concentration differences
4. Vector similarity
5. Transformation taxonomy
"""
import os
from typing import Optional

from config import BASE_DIR, DATA_PATH, FIGURE_PATH
from visualization import plot_cosine_similarity
from plot_cosine_similarity import generate_cosine_plots
from logging_utils import setup_logger, get_timestamp

def compare_canonical_harmonized(
    model_name: str = "vit-base-patch16-224",
    canonical_prefix: str = "vit_b16_canonical",
    harmonized_prefix: str = "vit_b16_harmonized",
    data_path: str = DATA_PATH,
    figure_path: str = FIGURE_PATH,
    log_file: Optional[str] = None
) -> None:
    """
    Generate comparison plots between canonical and harmonized models.
    
    Args:
        model_name: Base model name (e.g., "vit-base-patch16-224")
        canonical_prefix: Prefix for canonical model files
        harmonized_prefix: Prefix for harmonized model files
        data_path: Path to data directory
        figure_path: Path to save figures
        log_file: Path to log file
    """
    # Set up logger
    if log_file is None:
        timestamp = get_timestamp()
        log_dir = os.path.join(BASE_DIR, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"compare_models_{timestamp}.log")
        
    logger = setup_logger("compare_models", log_file)
    logger.info(f"Starting comparison between canonical and harmonized {model_name}")
    
    # Create model comparison directory
    comparison_dir = os.path.join(figure_path, "model_comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Generate cosine similarity plots
    logger.info("Generating cosine similarity plots")
    models = [model_name, model_name]
    prefixes = [canonical_prefix, harmonized_prefix]
    
    # Generate weighted and unweighted plots
    generate_cosine_plots(
        models,
        data_path,
        comparison_dir,
        generate_spectrum=False,  # Skip individual spectrum plots
        weights_paths=[None, None],
        file_prefixes=prefixes,
        log_file=log_file
    )
    
    # Create a special version of the weighted cosine plot specifically for the paper
    logger.info("Generating paper-ready cosine similarity plot")
    try:
        paper_plot_path = plot_cosine_similarity(
            models, 
            data_path, 
            comparison_dir, 
            weighted=True,
            weights_paths=[None, None],
            file_prefixes=prefixes,
            logger=logger
        )
        
        # Copy to transformation_by_layer.png for paper reference
        import shutil
        paper_file = os.path.join(comparison_dir, "transformation_by_layer.png")
        shutil.copy(paper_plot_path, paper_file)
        logger.info(f"Paper-ready plot saved to {paper_file}")
    except Exception as e:
        logger.error(f"Error generating paper-ready plot: {e}")
    
    logger.info("Model comparison complete")
    
    # Print instructions for further analyses
    logger.info("\nTo complete the full set of analyses for the paper, run:")
    logger.info("1. analyze_modes.py to generate KL divergence, concentration differences, and vector similarity")
    logger.info("2. compute_optimal_images.py to visualize the semantic modes")
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare canonical and harmonized ViT models")
    parser.add_argument("--model", type=str, default="vit-base-patch16-224", 
                        help="Base model name (e.g., 'vit-base-patch16-224')")
    parser.add_argument("--canonical_prefix", type=str, default="vit_b16_canonical", 
                        help="Prefix for canonical model files")
    parser.add_argument("--harmonized_prefix", type=str, default="vit_b16_harmonized", 
                        help="Prefix for harmonized model files")
    parser.add_argument("--data_path", type=str, default=DATA_PATH, 
                        help="Path to data directory")
    parser.add_argument("--figure_path", type=str, default=FIGURE_PATH, 
                        help="Path to save figures")
    
    args = parser.parse_args()
    
    compare_canonical_harmonized(
        args.model,
        args.canonical_prefix,
        args.harmonized_prefix,
        args.data_path,
        args.figure_path
    ) 