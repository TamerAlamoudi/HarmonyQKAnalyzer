#!/usr/bin/env python3
# compute_svd.py
"""
Script to compute Singular Value Decomposition on Q and K matrices
and save the resulting U, V, and S matrices.
"""
import os
import argparse
import json
import numpy as np
import torch
from typing import List, Optional, Dict

from config import BASE_DIR, DATA_PATH, DEFAULT_MODELS, get_device
from model_loader import load_model, get_model_name_for_files
from model_analysis import ModelAnalyzer
from logging_utils import setup_logger, get_timestamp

def compute_and_save_svd(
    model_names: List[str],
    data_path: str,
    device: torch.device,
    weights_paths: Optional[List[str]] = None,
    model_configs: Optional[List[Dict]] = None,
    file_prefixes: Optional[List[str]] = None,
    log_file: Optional[str] = None
) -> None:
    """
    Compute SVD for each model and save the results.
    
    Args:
        model_names: List of model names to process
        data_path: Path to save the results
        device: Device to use for computation
        weights_paths: List of paths to custom weights files (optional)
        model_configs: List of model configurations (optional)
        file_prefixes: List of file prefixes to use when saving files (optional)
        log_file: Path to log file
    """
    # Set up logger
    logger = setup_logger("compute_svd", log_file)
    logger.info(f"Starting SVD computation for {len(model_names)} models")
    logger.info(f"Using device: {device}")
    
    # Initialize weights_paths and model_configs if not provided
    if weights_paths is None:
        weights_paths = [None] * len(model_names)
    if model_configs is None:
        model_configs = [None] * len(model_names)
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
    
    if len(model_configs) != len(model_names):
        logger.warning(f"Number of model configs ({len(model_configs)}) does not match number of models ({len(model_names)})")
        # Pad with None if too short
        if len(model_configs) < len(model_names):
            model_configs.extend([None] * (len(model_names) - len(model_configs)))
        # Truncate if too long
        model_configs = model_configs[:len(model_names)]
    
    if len(file_prefixes) != len(model_names):
        logger.warning(f"Number of file prefixes ({len(file_prefixes)}) does not match number of models ({len(model_names)})")
        # Pad with None if too short
        if len(file_prefixes) < len(model_names):
            file_prefixes.extend([None] * (len(model_names) - len(file_prefixes)))
        # Truncate if too long
        file_prefixes = file_prefixes[:len(model_names)]
    
    for i, model_name in enumerate(model_names):
        weights_path = weights_paths[i]
        model_config = model_configs[i]
        file_prefix = file_prefixes[i]
        
        # Log model info
        if weights_path:
            logger.info(f"Processing model: {model_name} with custom weights from {weights_path}")
        else:
            logger.info(f"Processing model: {model_name} (pretrained from Huggingface)")
        
        try:
            # Load the model
            model, _ = load_model(model_name, device, weights_path, model_config)
            
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
                    if head % 3 == 0:  # Log every few heads to avoid too much output
                        logger.info(f"  Computing SVD for head {head}/{model.config.num_attention_heads-1} in layer {layer}")
                    
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
            safe_model_name = get_model_name_for_files(model_name, weights_path, file_prefix)
            np.save(os.path.join(output_dir, f"{safe_model_name}_U_total.npy"), U_total)
            np.save(os.path.join(output_dir, f"{safe_model_name}_Vt_total.npy"), Vt_total)
            np.save(os.path.join(output_dir, f"{safe_model_name}_S_total.npy"), S_total)
            
            # Print the filename being used so users can reference it later
            logger.info(f"Files saved with prefix: {safe_model_name}")
            
            # If custom weights were used, save a mapping file for future reference
            if weights_path:
                mapping_file = os.path.join(output_dir, "custom_models_mapping.json")
                mapping = {}
                
                # Load existing mapping if it exists
                if os.path.exists(mapping_file):
                    with open(mapping_file, 'r') as f:
                        try:
                            mapping = json.load(f)
                        except json.JSONDecodeError:
                            logger.warning(f"Error loading existing mapping file: {mapping_file}")
                            mapping = {}
                
                # Add or update this model's entry
                mapping[safe_model_name] = {
                    "original_name": model_name,
                    "weights_path": weights_path,
                    "config": model_config
                }
                
                # Save the mapping
                with open(mapping_file, 'w') as f:
                    json.dump(mapping, f, indent=2)
            
            logger.info(f"Completed processing for {model_name}")
            
        except Exception as e:
            logger.error(f"Error processing {model_name}: {e}")
    
    logger.info("SVD computation completed")

def main():
    """Main function to parse arguments and run SVD computation."""
    parser = argparse.ArgumentParser(description="Compute SVD for transformer models")
    parser.add_argument("--models", nargs="+", help="List of model names to process")
    
    # Arguments for custom models
    parser.add_argument("--weights_paths", nargs="+", help="List of paths to custom weights files")
    parser.add_argument("--config_files", nargs="+", help="List of paths to model config JSON files")
    parser.add_argument("--file_prefixes", nargs="+", help="List of file prefixes to use when saving files")
    
    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="Path to save the results")
    parser.add_argument("--log_dir", type=str, default=os.path.join(BASE_DIR, "logs"), help="Directory for log files")
    args = parser.parse_args()
    
    # Use provided models or default ones
    model_names = args.models if args.models else DEFAULT_MODELS
    
    # Process config files if provided
    model_configs = None
    if args.config_files:
        model_configs = []
        for config_file in args.config_files:
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    model_configs.append(config)
            except Exception as e:
                print(f"Error loading config file {config_file}: {e}")
                model_configs.append(None)
    
    # Create log directory if it doesn't exist
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Generate log file path
    timestamp = get_timestamp()
    log_file = os.path.join(args.log_dir, f"compute_svd_{timestamp}.log")
    
    # Set up a logger for the main function
    logger = setup_logger("main", log_file)
    
    # Run SVD computation
    compute_and_save_svd(
        model_names, 
        args.data_path, 
        get_device(), 
        args.weights_paths, 
        model_configs,
        args.file_prefixes,
        log_file
    )

if __name__ == "__main__":
    main() 