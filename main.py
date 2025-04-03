#!/usr/bin/env python3
# main.py
"""
Main script to run the complete DissectingViT analysis pipeline.
"""
import argparse
import os
from typing import List, Optional

from compute_optimal_images import compute_activations, visualize_optimal_images
from compute_svd import compute_and_save_svd
from logging_utils import get_timestamp, setup_logger
from plot_cosine_similarity import generate_cosine_plots

from config import BASE_DIR, DATA_PATH, DEFAULT_MODELS, FIGURE_PATH, get_device


def run_pipeline(
    model_names: List[str],
    data_path: str,
    figure_path: str,
    dataset_path: str,
    run_svd: bool = True,
    run_cosine: bool = True,
    run_activations: bool = True,
    run_visualizations: bool = True,
    visualization_config: Optional[dict] = None,
    log_file: Optional[str] = None,
) -> None:
    """
    Run the complete DissectingViT analysis pipeline.

    Args:
        model_names: List of model names to process
        data_path: Path to the data directory
        figure_path: Path to save the output figures
        dataset_path: Path to the dataset
        run_svd: Whether to run SVD computation
        run_cosine: Whether to generate cosine similarity plots
        run_activations: Whether to compute activations
        run_visualizations: Whether to generate visualizations
        visualization_config: Configuration for visualizations
        log_file: Path to log file
    """
    # Set up logger
    logger = setup_logger("pipeline", log_file)
    logger.info(f"Starting DissectingViT pipeline for {len(model_names)} models")

    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")

    # Step 1: Compute SVD
    if run_svd:
        logger.info("=== Step 1: Computing SVD ===")
        compute_and_save_svd(model_names, data_path, device, log_file)

    # Step 2: Generate cosine similarity plots
    if run_cosine:
        logger.info("=== Step 2: Generating cosine similarity plots ===")
        generate_cosine_plots(model_names, data_path, figure_path, True, log_file)

    # Step 3: Compute activations
    if run_activations:
        logger.info("=== Step 3: Computing activations ===")
        for model_name in model_names:
            logger.info(f"Computing activations for {model_name}")
            try:
                compute_activations(
                    model_name,
                    dataset_path,
                    os.path.join(data_path, "optimal_images"),
                    device=device,
                    log_file=log_file,
                )
            except Exception as e:
                logger.error(f"Error computing activations for {model_name}: {e}")

    # Step 4: Generate visualizations
    if run_visualizations:
        logger.info("=== Step 4: Generating visualizations ===")

        # Use default visualization config if none provided
        if visualization_config is None:
            visualization_config = {
                "layer": 0,
                "head": 0,
                "sign": "p",
                "modes": [0, 1, 2, 3, 4],
            }

        for model_name in model_names:
            logger.info(f"Generating visualizations for {model_name}")
            try:
                visualize_optimal_images(
                    model_name,
                    dataset_path,
                    figure_path,
                    data_path,
                    visualization_config.get("layer", 0),
                    visualization_config.get("head", 0),
                    visualization_config.get("sign", "p"),
                    visualization_config.get("modes", [0, 1, 2, 3, 4]),
                    device=device,
                    log_file=log_file,
                )
            except Exception as e:
                logger.error(f"Error generating visualizations for {model_name}: {e}")

    logger.info("Pipeline completed successfully")


def main():
    """Main function to parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(description="Run DissectingViT analysis pipeline")
    parser.add_argument("--models", nargs="+", help="List of model names to process")
    parser.add_argument(
        "--data_path", type=str, default=DATA_PATH, help="Path to the data directory"
    )
    parser.add_argument(
        "--figure_path",
        type=str,
        default=FIGURE_PATH,
        help="Path to save the output figures",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=os.path.join(BASE_DIR, "datasets/imagenette2/val/"),
        help="Path to the dataset",
    )
    parser.add_argument("--no_svd", action="store_true", help="Skip SVD computation")
    parser.add_argument(
        "--no_cosine", action="store_true", help="Skip cosine similarity plots"
    )
    parser.add_argument(
        "--no_activations", action="store_true", help="Skip activation computation"
    )
    parser.add_argument(
        "--no_visualizations", action="store_true", help="Skip visualizations"
    )
    parser.add_argument(
        "--layer", type=int, default=0, help="Layer index for visualizations"
    )
    parser.add_argument(
        "--head", type=int, default=0, help="Head index for visualizations"
    )
    parser.add_argument(
        "--sign",
        type=str,
        default="p",
        choices=["p", "n"],
        help="Sign of the activations ('p' or 'n')",
    )
    parser.add_argument(
        "--modes",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="List of mode indices to visualize",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=os.path.join(BASE_DIR, "logs"),
        help="Directory for log files",
    )
    args = parser.parse_args()

    # Use provided models or default ones
    model_names = args.models if args.models else DEFAULT_MODELS

    # Create log directory if it doesn't exist
    os.makedirs(args.log_dir, exist_ok=True)

    # Generate log file path
    timestamp = get_timestamp()
    log_file = os.path.join(args.log_dir, f"pipeline_{timestamp}.log")

    # Create visualization config
    visualization_config = {
        "layer": args.layer,
        "head": args.head,
        "sign": args.sign,
        "modes": args.modes,
    }

    # Run pipeline
    run_pipeline(
        model_names,
        args.data_path,
        args.figure_path,
        args.dataset_path,
        not args.no_svd,
        not args.no_cosine,
        not args.no_activations,
        not args.no_visualizations,
        visualization_config,
        log_file,
    )


if __name__ == "__main__":
    main()
