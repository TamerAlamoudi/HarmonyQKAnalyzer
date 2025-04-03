#!/usr/bin/env python3
# compute_optimal_images.py
"""
Script to compute activations and find optimal images for different modes.
"""
import argparse
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from data_utils import ImageDataset, load_pickle, save_numpy_array
from logging_utils import get_timestamp, setup_logger
from model_loader import create_hook_model, get_start_index, load_model
from torch.utils.data import DataLoader

from config import BASE_DIR, DATA_PATH, FIGURE_PATH, IMAGENET_VAL_PATH, get_device


def compute_activations(
    model_name: str,
    dataset_path: str,
    output_dir: str,
    batch_size: int = 128,
    device: torch.device = torch.device("cpu"),
    log_file: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Compute activations for all layers and heads, and save them.

    Args:
        model_name: Name of the model to use
        dataset_path: Path to the dataset
        output_dir: Directory to save the output files
        batch_size: Batch size for dataloader
        device: Device to use for computation
        log_file: Path to log file

    Returns:
        Tuple containing paths to the positive and negative activation files
    """
    # Set up logger
    logger = setup_logger("compute_activations", log_file)
    logger.info(f"Starting activation computation for {model_name}")
    logger.info(f"Using device: {device}")

    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load the model and processor
        logger.info(f"Loading model {model_name}")
        model, processor = load_model(model_name, device)

        # Create hook model
        hook_model = create_hook_model(model, model_name)

        # Get start index based on model type
        start_index = get_start_index(model_name)

        # Load image list
        logger.info("Loading image list")
        image_list_path = os.path.join(
            BASE_DIR, "datasets", "imagenette2", "imagenette_list_val.pkl"
        )
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
        U_path = os.path.join(DATA_PATH, "UVS", f"{model_name}_U_total.npy")
        Vt_path = os.path.join(DATA_PATH, "UVS", f"{model_name}_Vt_total.npy")

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

        logger.info(
            f"Processing {n_layer} layers, {n_head} heads, {n_mode} modes, {n_samples} samples"
        )

        # Pre-allocate arrays for activations
        activations_p = np.zeros((n_layer, n_head, n_mode, n_samples))
        activations_n = np.zeros((n_layer, n_head, n_mode, n_samples))

        # Process batches
        sample_idx = 0
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 10 == 0:
                logger.info(f"Processing batch {batch_idx}/{len(dataloader)}")

            # Move batch to device
            batch = batch.to(device)

            # Get batch size
            current_batch_size = batch.size(0)

            # Get hidden states
            with torch.no_grad():
                outputs = hook_model(
                    batch, output_hidden_states=False, output_attentions=False
                )
                hidden_states = hook_model.outputs

            # Process each layer and head
            for layer in range(n_layer):
                layer_states = hidden_states[layer]
                layer_states = layer_states[:, start_index:, :]  # Remove CLS token

                # Compute activations for all heads and modes
                for head in range(n_head):
                    for mode in range(n_mode):
                        # Get U and V vectors
                        U = (
                            torch.from_numpy(U_total[layer, head, :, mode])
                            .float()
                            .to(device)
                        )
                        V = (
                            torch.from_numpy(Vt_total[layer, head, mode, :])
                            .float()
                            .to(device)
                            .T
                        )

                        # Compute activations for positive sign
                        U_activation = torch.matmul(layer_states, U)
                        V_activation = torch.matmul(layer_states, V)
                        activations_p[
                            layer,
                            head,
                            mode,
                            sample_idx : sample_idx + current_batch_size,
                        ] = (
                            torch.sum(U_activation * V_activation, dim=1).cpu().numpy()
                        )

                        # Compute activations for negative sign
                        U_neg = -U
                        V_neg = -V
                        U_activation_neg = torch.matmul(layer_states, U_neg)
                        V_activation_neg = torch.matmul(layer_states, V_neg)
                        activations_n[
                            layer,
                            head,
                            mode,
                            sample_idx : sample_idx + current_batch_size,
                        ] = (
                            torch.sum(U_activation_neg * V_activation_neg, dim=1)
                            .cpu()
                            .numpy()
                        )

            # Update sample index
            sample_idx += current_batch_size

            # Clear hook outputs to free memory
            hook_model.clear_outputs()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Save activations
        logger.info("Saving activations")
        activations_p_path = os.path.join(output_dir, f"{model_name}_activations_p.npy")
        activations_n_path = os.path.join(output_dir, f"{model_name}_activations_n.npy")

        save_numpy_array(activations_p, activations_p_path)
        save_numpy_array(activations_n, activations_n_path)

        logger.info(
            f"Activations saved to {activations_p_path} and {activations_n_path}"
        )

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
    device: torch.device = torch.device("cpu"),
    log_file: Optional[str] = None,
) -> str:
    """
    Visualize optimal images for specified modes.

    Args:
        model_name: Name of the model to use
        dataset_path: Path to the dataset
        figure_path: Path to save the output figures
        data_path: Path to the data directory containing UVS and activations files
        layer: Layer index
        head: Head index
        sign: Sign of the activations ('p' or 'n')
        modes: List of mode indices to visualize
        device: Device to use for computation
        log_file: Path to log file

    Returns:
        Path to the saved visualization
    """
    from visualization import plot_mode_heatmap

    # Set up logger
    logger = setup_logger("visualize_optimal_images", log_file)
    logger.info(
        f"Starting visualization for {model_name}, Layer {layer}, Head {head}, Sign {sign}"
    )

    try:
        # Load model and processor
        logger.info(f"Loading model {model_name}")
        model, processor = load_model(model_name, device)

        # Create hook model
        hook_model = create_hook_model(model, model_name)

        # Load image list
        logger.info("Loading image list")
        image_list_path = os.path.join(
            BASE_DIR, "datasets", "imagenette2", "imagenette_list_val.pkl"
        )
        im_list = load_pickle(image_list_path)

        # Load U and Vt matrices
        logger.info("Loading U and Vt matrices")
        U_path = os.path.join(data_path, "UVS", f"{model_name}_U_total.npy")
        Vt_path = os.path.join(data_path, "UVS", f"{model_name}_Vt_total.npy")

        U_total = np.load(U_path)
        Vt_total = np.load(Vt_path)

        # Load activations
        logger.info("Loading activations")
        activations_p_path = os.path.join(
            data_path, "optimal_images", f"{model_name}_activations_p.npy"
        )
        activations_n_path = os.path.join(
            data_path, "optimal_images", f"{model_name}_activations_n.npy"
        )

        # Check if activations exist
        if not os.path.exists(activations_p_path) or not os.path.exists(
            activations_n_path
        ):
            logger.info("Activations not found, computing them now")
            activations_p_path, activations_n_path = compute_activations(
                model_name,
                dataset_path,
                os.path.join(data_path, "optimal_images"),
                device=device,
                log_file=log_file,
            )

        activations_p = np.load(activations_p_path)
        activations_n = np.load(activations_n_path)

        # Create mode list
        mode_list = [(layer, head, mode, sign) for mode in modes]

        # Create visualization
        logger.info(f"Creating visualization for {len(mode_list)} modes")
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
            file_name=f"{model_name}_L{layer}_H{head}_{sign}",
            imagenet_val_path=dataset_path,
            start_index=get_start_index(model_name),
            im_per_mode=8,
            dpi=150,
            logger=logger,
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
    parser = argparse.ArgumentParser(
        description="Compute and visualize optimal images for modes"
    )
    parser.add_argument("--model", type=str, required=True, help="Model name to use")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=IMAGENET_VAL_PATH,
        help="Path to the dataset",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=DATA_PATH,
        help="Path to the data directory containing UVS files",
    )
    parser.add_argument(
        "--figure_path",
        type=str,
        default=FIGURE_PATH,
        help="Path to save the output figures",
    )
    parser.add_argument("--layer", type=int, default=0, help="Layer index")
    parser.add_argument("--head", type=int, default=0, help="Head index")
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
        "--compute_only",
        action="store_true",
        help="Only compute activations, don't visualize",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=os.path.join(BASE_DIR, "logs"),
        help="Directory for log files",
    )
    args = parser.parse_args()

    # Create log directory if it doesn't exist
    os.makedirs(args.log_dir, exist_ok=True)

    # Generate log file path
    timestamp = get_timestamp()
    log_file = os.path.join(args.log_dir, f"optimal_images_{timestamp}.log")

    # Determine device
    device = get_device()

    if args.compute_only:
        # Compute activations only
        compute_activations(
            args.model,
            args.dataset_path,
            os.path.join(args.data_path, "optimal_images"),
            device=device,
            log_file=log_file,
        )
    else:
        # Visualize optimal images
        visualize_optimal_images(
            args.model,
            args.dataset_path,
            args.figure_path,
            args.data_path,
            args.layer,
            args.head,
            args.sign,
            args.modes,
            device=device,
            log_file=log_file,
        )


if __name__ == "__main__":
    main()
