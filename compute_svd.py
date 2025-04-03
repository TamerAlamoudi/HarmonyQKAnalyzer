#!/usr/bin/env python3
# compute_svd.py
"""
Script to compute Singular Value Decomposition on Q and K matrices
and save the resulting U, V, and S matrices.
"""
import argparse
import os
from typing import List, Optional

import numpy as np
import torch
from logging_utils import get_timestamp, setup_logger
from model_analysis import ModelAnalyzer
from model_loader import load_model

from config import BASE_DIR, DATA_PATH, DEFAULT_MODELS, get_device


def compute_and_save_svd(
    model_names: List[str],
    data_path: str,
    device: torch.device,
    log_file: Optional[str] = None,
) -> None:
    """
    Compute SVD for each model and save the results.

    Args:
        model_names: List of model names to process
        data_path: Path to save the results
        device: Device to use for computation
        log_file: Path to log file
    """
    # Set up logger
    logger = setup_logger("compute_svd", log_file)
    logger.info(f"Starting SVD computation for {len(model_names)} models")
    logger.info(f"Using device: {device}")

    for model_name in model_names:
        logger.info(f"Processing model: {model_name}")

        try:
            # Load the model
            model, _ = load_model(model_name, device)

            # Create analyzer
            analyzer = ModelAnalyzer(model)

            # Get Q and K matrices
            logger.info(f"Extracting Q and K matrices for {model_name}")
            QK = analyzer.get_QK(model.config.num_attention_heads)

            # Initialize arrays to store results
            U_total = []  # Left singular vectors
            Vt_total = []  # Right singular vectors (transposed)
            S_total = []  # Singular values

            # Process each layer
            for layer in range(model.config.num_hidden_layers):
                logger.info(f"Computing SVD for layer {layer}")
                Q, K, _ = QK[layer]

                U_head = []
                Vt_head = []
                S_head = []

                # Process each head
                for head in range(model.config.num_attention_heads):
                    # Compute interaction matrix
                    M = np.zeros((Q.shape[0], Q.shape[0]))
                    M[:, :] = np.matmul(Q[:, head, :], K[:, head, :].T)

                    # Compute SVD
                    U, S, Vt, _ = analyzer.svd_angles(M[:, :])

                    # Store results
                    U_head.append(U)
                    Vt_head.append(Vt)
                    S_head.append(S)

                U_total.append(U_head)
                Vt_total.append(Vt_head)
                S_total.append(S_head)

            # Convert to numpy arrays and limit to embedding size
            embedding_size = Q.shape[2]
            U_total = np.array(U_total)[:, :, :, :embedding_size]
            Vt_total = np.array(Vt_total)[:, :, :embedding_size, :]
            S_total = np.array(S_total)[:, :, :embedding_size]

            # Create output directory if it doesn't exist
            output_dir = os.path.join(data_path, "UVS")
            os.makedirs(output_dir, exist_ok=True)

            # Save the arrays
            logger.info(f"Saving results for {model_name}")
            np.save(os.path.join(output_dir, f"{model_name}_U_total.npy"), U_total)
            np.save(os.path.join(output_dir, f"{model_name}_Vt_total.npy"), Vt_total)
            np.save(os.path.join(output_dir, f"{model_name}_S_total.npy"), S_total)

            logger.info(f"Completed processing for {model_name}")

        except Exception as e:
            logger.error(f"Error processing {model_name}: {e}")

    logger.info("SVD computation completed")


def main():
    """Main function to parse arguments and run SVD computation."""
    parser = argparse.ArgumentParser(description="Compute SVD for transformer models")
    parser.add_argument("--models", nargs="+", help="List of model names to process")
    parser.add_argument(
        "--data_path", type=str, default=DATA_PATH, help="Path to save the results"
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
    log_file = os.path.join(args.log_dir, f"compute_svd_{timestamp}.log")

    # Run SVD computation
    compute_and_save_svd(model_names, args.data_path, get_device(), log_file)


if __name__ == "__main__":
    main()
