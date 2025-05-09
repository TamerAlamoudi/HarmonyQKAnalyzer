#!/usr/bin/env python3
# compute_optimal_images.py
"""
Script to compute activations and find optimal images for different modes.
"""
import os
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple, Optional
import time

from config import BASE_DIR, DATA_PATH, FIGURE_PATH, IMAGENET_VAL_PATH, IMAGENET_LIST_PATH, get_device
from model_loader import load_model, create_hook_model, get_start_index, get_model_name_for_files
from data_utils import ImageDataset, load_pickle, save_numpy_array
from logging_utils import setup_logger, get_timestamp
from visualization import plot_mode_heatmap

def compute_activations(
    model_name: str,
    dataset_path: str,
    output_dir: str,
    weights_path: Optional[str] = None,
    model_config: Optional[Dict] = None,
    batch_size: int = 128,
    device: torch.device = torch.device('cpu'),
    log_file: Optional[str] = None,
    file_prefix: Optional[str] = None
) -> Tuple[str, str]:
    """
    Compute activations for all layers and heads, and save them.
    
    Args:
        model_name: Name of the model to use
        dataset_path: Path to the dataset
        output_dir: Directory to save the output files
        weights_path: Path to custom weights file (optional)
        model_config: Model configuration dictionary (optional)
        batch_size: Batch size for dataloader
        device: Device to use for computation
        log_file: Path to log file
        file_prefix: Explicitly specify the file prefix for loading SVD files
        
    Returns:
        Tuple containing paths to the positive and negative activation files
    """
    # Set up logger
    logger = setup_logger("compute_activations", log_file)
    
    if weights_path:
        logger.info(f"Starting activation computation for {model_name} with custom weights from {weights_path}")
    else:
        logger.info(f"Starting activation computation for {model_name} (pretrained from Huggingface)")
        
    logger.info(f"Using device: {device}")
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the model and processor
        logger.info(f"Loading model {model_name}")
        model, processor = load_model(model_name, device, weights_path, model_config)
        
        # Create hook model
        hook_model = create_hook_model(model, model_name)
        
        # Get start index based on model type
        start_index = get_start_index(model_name)
        
        # Load image list
        logger.info("Loading image list")
        # image_list_path = os.path.join(BASE_DIR, "datasets", "imagenette2", "imagenette_list_val.pkl")
        image_list_path = IMAGENET_LIST_PATH
        try:
            im_list = load_pickle(image_list_path)
            logger.info(f"Loaded {len(im_list)} images from list")
        except Exception as e:
            logger.error(f"Error loading image list: {e}")
            raise
        
        # Create dataset and dataloader
        logger.info("Creating dataset and dataloader")
        dataset = ImageDataset(im_list, processor, dataset_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Load U and Vt matrices
        logger.info("Loading U and Vt matrices")
        safe_model_name = get_model_name_for_files(model_name, weights_path, file_prefix)
        U_path = os.path.join(DATA_PATH, "UVS", f"{safe_model_name}_U_total.npy")
        Vt_path = os.path.join(DATA_PATH, "UVS", f"{safe_model_name}_Vt_total.npy")
        
        try:
            U_total = np.load(U_path)
            Vt_total = np.load(Vt_path)
            logger.info(f"Loaded U shape: {U_total.shape}, Vt shape: {Vt_total.shape}")
        except Exception as e:
            logger.error(f"Error loading U and Vt matrices: {e}")
            raise
        
        # Get dimensions
        n_layer = U_total.shape[0]
        n_head = U_total.shape[1]
        n_mode = U_total.shape[3]
        n_samples = len(im_list)
        
        logger.info(f"Processing {n_layer} layers, {n_head} heads, {n_mode} modes, {n_samples} samples")
        
        # Allocate arrays for activations
        activations_p = np.zeros((n_layer, n_head, n_mode, n_samples))
        activations_n = np.zeros((n_layer, n_head, n_mode, n_samples))
        
        # Process batches
        sample_idx = 0
        total_batches = len(dataloader)
        start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 5 == 0 or batch_idx == total_batches - 1:
                progress_pct = (batch_idx + 1) / total_batches * 100
                elapsed_time = time.time() - start_time
                
                # Estimate remaining time if we're not at the start
                if batch_idx > 0:
                    time_per_batch = elapsed_time / (batch_idx + 1)
                    remaining_batches = total_batches - (batch_idx + 1)
                    estimated_time = time_per_batch * remaining_batches
                    time_info = f", est. remaining: {estimated_time:.1f}s"
                else:
                    time_info = ""
                
                logger.info(f"Processing batch {batch_idx+1}/{total_batches} ({progress_pct:.1f}%){time_info}")
            
            # Move batch to device
            batch = batch.to(device)
            
            # Get batch size
            current_batch_size = batch.size(0)
            
            # Get hidden states
            with torch.no_grad():
                outputs = hook_model(batch, output_hidden_states=False, output_attentions=False)
                hidden_states = hook_model.outputs
            
            # Process each layer and head
            for layer in range(n_layer):
                logger.info(f"Processing layer {layer}/{n_layer-1}")
                layer_states = hidden_states[layer]
                layer_states = layer_states[:, start_index:, :]  # Remove CLS token
                
                # Compute activations for all heads and modes
                for head in range(n_head):
                    if head % 3 == 0:  # Log every few heads to avoid too much output
                        logger.info(f"  Processing head {head}/{n_head-1} for layer {layer}")
                    
                    for mode in range(n_mode):
                        # Get U and V vectors
                        U = torch.from_numpy(U_total[layer, head, :, mode]).float().to(device)
                        V = torch.from_numpy(Vt_total[layer, head, mode, :]).float().to(device).T
                        
                        # Compute activations for positive sign
                        U_activation = torch.matmul(layer_states, U)
                        V_activation = torch.matmul(layer_states, V)
                        activations_p[layer, head, mode, sample_idx:sample_idx+current_batch_size] = (
                            torch.sum(U_activation * V_activation, dim=1).cpu().numpy()
                        )
                        
                        # Compute activations for negative sign
                        U_neg = -U
                        V_neg = -V
                        U_activation_neg = torch.matmul(layer_states, U_neg)
                        V_activation_neg = torch.matmul(layer_states, V_neg)
                        activations_n[layer, head, mode, sample_idx:sample_idx+current_batch_size] = (
                            torch.sum(U_activation_neg * V_activation_neg, dim=1).cpu().numpy()
                        )
            
            # Update sample index
            sample_idx += current_batch_size
            
            # Clear hook outputs to free memory
            hook_model.clear_outputs()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Save activations
        logger.info("Saving activations")
        safe_model_name = get_model_name_for_files(model_name, weights_path)
        activations_p_path = os.path.join(output_dir, f"{safe_model_name}_activations_p.npy")
        activations_n_path = os.path.join(output_dir, f"{safe_model_name}_activations_n.npy")
        
        save_numpy_array(activations_p, activations_p_path)
        save_numpy_array(activations_n, activations_n_path)
        
        logger.info(f"Activations saved to {activations_p_path} and {activations_n_path}")
        
        # Clean up
        hook_model.remove_hooks()
        del hook_model, model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return activations_p_path, activations_n_path
    
    except Exception as e:
        logger.error(f"Error computing activations: {e}")
        raise

def visualize_optimal_images(
    model_name: str,
    dataset_path: str,
    figure_path: str,
    data_path: str,
    layer: int = 0,
    head: int = 0,
    sign: str = "p",
    modes: List[int] = [0, 1, 2, 3, 4],
    weights_path: Optional[str] = None,
    model_config: Optional[Dict] = None,
    device: torch.device = torch.device('cpu'),
    log_file: Optional[str] = None,
    file_prefix: Optional[str] = None,
    save_diagnostics: bool = False
) -> str:
    """
    Visualize optimal images for specific modes.
    
    Args:
        model_name: Name of the model to use
        dataset_path: Path to the dataset
        figure_path: Path to save the output figures
        data_path: Path to the data directory containing UVS files
        layer: Layer index to visualize
        head: Head index to visualize
        sign: Sign of the activations ('p' or 'n')
        modes: List of mode indices to visualize
        weights_path: Path to custom weights file (optional)
        model_config: Model configuration dictionary (optional)
        device: Device to use for computation
        log_file: Path to log file
        file_prefix: Explicitly specify the file prefix for loading SVD files
        save_diagnostics: Whether to save additional diagnostic information
        
    Returns:
        Path to the output figure directory
    """
    # Set up logger
    logger = setup_logger("visualize_optimal_images", log_file)
    
    # Get safe model name for file paths
    safe_model_name = get_model_name_for_files(model_name, weights_path, file_prefix)
    
    logger.info(f"Starting visualization for {safe_model_name} ({'pretrained' if weights_path is None else 'custom weights'}), Layer {layer}, Head {head}, Sign {sign}")
    
    try:
        # Create a more descriptive file name
        if model_name.startswith('google/'):
            model_short = model_name.split('/')[-1]
        else:
            model_short = model_name.replace('/', '_')
            
        # Add a suffix for custom models
        if weights_path:
            weights_filename = os.path.basename(weights_path)
            weights_name = os.path.splitext(weights_filename)[0]
            model_short = f"{model_short}_custom_{weights_name}"

        # Create output directory
        output_dir = os.path.join(figure_path, "optimal_images", safe_model_name, f"layer_{layer}_head_{head}_sign_{sign}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model and processor for visualizing attention maps
        logger.info(f"Loading model {model_name}")
        model, processor = load_model(model_name, device, weights_path, model_config)
        
        # Create hook model
        hook_model = create_hook_model(model, model_name)
        
        # Load image list
        logger.info("Loading image list")
        # image_list_path = os.path.join(BASE_DIR, "datasets", "imagenette2", "imagenette_list_val.pkl")
        image_list_path = IMAGENET_LIST_PATH
        try:
            im_list = load_pickle(image_list_path)
            logger.info(f"Loaded {len(im_list)} images from list")
        except Exception as e:
            logger.error(f"Error loading image list: {e}")
            raise


        # Load U and Vt matrices
        logger.info("Loading U and Vt matrices")
        U_path = os.path.join(data_path, "UVS", f"{safe_model_name}_U_total.npy")
        Vt_path = os.path.join(data_path, "UVS", f"{safe_model_name}_Vt_total.npy")
        
        try:
            U_total = np.load(U_path)
            Vt_total = np.load(Vt_path)
            logger.info(f"Loaded U shape: {U_total.shape}, Vt shape: {Vt_total.shape}")
        except Exception as e:
            logger.error(f"Error loading U and Vt matrices: {e}")
            raise
        
        # Load activations
        logger.info("Loading activations")
        activations_p_file = os.path.join(data_path, "optimal_images", f"google_{model_short}_activations_p.npy")
        activations_n_file = os.path.join(data_path, "optimal_images", f"google_{model_short}_activations_n.npy")
        try:
            activations_p = np.load(activations_p_file)
            activations_n = np.load(activations_n_file)
            logger.info(f"Loaded activations shapes: p={activations_p.shape}, n={activations_n.shape}")
        except Exception as e:
            logger.error(f"Error loading activations: {e}")
            logger.error("Please run with --compute_only first")
            raise
        
        # Create mode list
        mode_list = [(layer, head, mode, sign) for mode in modes]
        
        # Create visualization
        logger.info(f"Creating visualization for {len(mode_list)} modes")
        
        
        file_name = f"{model_short}_L{layer}_H{head}_{sign}"
        
        output_path = plot_mode_heatmap(
            mode_list=mode_list,
            im_list=im_list,
            U_total=U_total,
            Vt_total=Vt_total,
            activations_p=activations_p,
            activations_n=activations_n,
            model_name=model_name,
            hook_model=hook_model,
            processor=processor,
            output_dir=os.path.join(figure_path, "optimal_images"),
            file_name=file_name,
            imagenet_val_path=dataset_path,
            start_index=get_start_index(model_name),
            im_per_mode=8,
            dpi=150,
            logger=logger,
            save_diagnostics=save_diagnostics
        )
        
        logger.info(f"Visualization saved to {output_path}")
        
        # Clean up
        hook_model.remove_hooks()
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error in visualization: {e}")
        raise

def main():
    """Main function to parse arguments and run visualization."""
    parser = argparse.ArgumentParser(description="Compute and visualize optimal images for modes")
    parser.add_argument("--model", type=str, required=True, help="Model name to use")
    parser.add_argument("--dataset_path", type=str, default=IMAGENET_VAL_PATH, help="Path to the dataset")
    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="Path to the data directory containing UVS files")
    parser.add_argument("--figure_path", type=str, default=FIGURE_PATH, help="Path to save the output figures")
    
    # Arguments for custom models
    parser.add_argument("--weights_path", type=str, help="Path to custom weights file")
    parser.add_argument("--config_file", type=str, help="Path to model config JSON file")
    parser.add_argument("--file_prefix", type=str, help="Explicitly specify the file prefix for loading SVD files")
    
    # Visualization parameters
    parser.add_argument("--layer", type=int, help="Layer index (if not specified, process all layers)")
    parser.add_argument("--head", type=int, help="Head index (if not specified, process all heads)")
    parser.add_argument("--sign", type=str, choices=["p", "n"], help="Sign of the activations ('p' or 'n') (if not specified, process both)")
    parser.add_argument("--modes", type=int, nargs="+", help="List of mode indices to visualize (if not specified, use top 20 modes)")
    parser.add_argument("--top_modes", type=int, default=20, help="Number of top modes to use when no specific modes are provided")
    
    parser.add_argument("--compute_only", action="store_true", help="Only compute activations, don't visualize")
    parser.add_argument("--save_diagnostics", action="store_true", help="Save additional diagnostic information about modes")
    parser.add_argument("--log_dir", type=str, default=os.path.join(BASE_DIR, "logs"), help="Directory for log files")
    args = parser.parse_args()
    
    # Create log directory if it doesn't exist
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Generate log file path
    timestamp = get_timestamp()
    log_file = os.path.join(args.log_dir, f"optimal_images_{timestamp}.log")
    
    # Determine device
    device = get_device()
    
    # Load model config if provided
    model_config = None
    if args.config_file:
        try:
            with open(args.config_file, 'r') as f:
                model_config = json.load(f)
        except Exception as e:
            print(f"Error loading config file {args.config_file}: {e}")
    
    if args.compute_only:
        # Compute activations only
        compute_activations(
            args.model,
            args.dataset_path,
            os.path.join(args.data_path, "optimal_images"),
            args.weights_path,
            model_config,
            device=device,
            log_file=log_file,
            file_prefix=args.file_prefix
        )
    else:
        # First, load U matrices to get the dimensions
        logger = setup_logger("main", log_file)
        logger.info("Loading U matrix to determine dimensions")
        
        safe_model_name = get_model_name_for_files(args.model, args.weights_path, args.file_prefix)
        U_path = os.path.join(args.data_path, "UVS", f"{safe_model_name}_U_total.npy")
        try:
            U_total = np.load(U_path)
            n_layers = U_total.shape[0]
            n_heads = U_total.shape[1]
            n_modes = U_total.shape[3]
            logger.info(f"Found {n_layers} layers, {n_heads} heads, {n_modes} modes")
        except Exception as e:
            logger.error(f"Error loading U matrix: {e}")
            logger.error("Please run compute_svd.py first")
            raise
        
        # Determine layers to process
        layers = [args.layer] if args.layer is not None else list(range(n_layers))
        logger.info(f"Processing layers: {layers}")
        
        # Determine heads to process
        heads = [args.head] if args.head is not None else list(range(n_heads))
        logger.info(f"Processing heads: {heads}")
        
        # Determine signs to process
        signs = [args.sign] if args.sign is not None else ["p", "n"]
        logger.info(f"Processing signs: {signs}")
        
        # Determine modes to process
        if args.modes is not None:
            modes = args.modes
            logger.info(f"Using specified modes: {modes}")
        else:
            modes = list(range(min(args.top_modes, n_modes)))
            logger.info(f"Using top {len(modes)} modes: {modes}")
        
        # Process each combination
        for layer in layers:
            for head in heads:
                for sign in signs:
                    logger.info(f"Visualizing for layer {layer}, head {head}, sign {sign}")
                    visualize_optimal_images(
                        args.model,
                        args.dataset_path,
                        args.figure_path,
                        args.data_path,
                        layer,
                        head,
                        sign,
                        modes,
                        args.weights_path,
                        model_config,
                        device=device,
                        log_file=log_file,
                        file_prefix=args.file_prefix,
                        save_diagnostics=args.save_diagnostics
                    )

if __name__ == "__main__":
    main() 