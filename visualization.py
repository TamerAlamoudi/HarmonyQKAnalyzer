# utils/visualization.py
"""
Visualization utilities for DissectingViT.
"""
import logging
import os
from typing import Any, Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def plot_mode_heatmap(
    mode_list: List[Tuple[int, int, int, str]],
    im_list: List[str],
    U_total: np.ndarray,
    Vt_total: np.ndarray,
    activations_p: np.ndarray,
    activations_n: np.ndarray,
    model_name: str,
    hook_model: Any,
    processor: Callable,
    output_dir: str,
    file_name: str,
    imagenet_val_path: str,
    start_index: int = 1,
    im_per_mode: int = 8,
    dpi: int = 200,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Plot heatmaps for specified modes.

    Args:
        mode_list: List of (layer, head, mode, sign) tuples
        im_list: List of image filenames
        U_total: U matrix from SVD
        Vt_total: V^T matrix from SVD
        activations_p: Activation values for positive sign
        activations_n: Activation values for negative sign
        model_name: Name of the model
        hook_model: Hook model for getting hidden states
        processor: Image processor function
        output_dir: Output directory
        file_name: Output file name
        imagenet_val_path: Path to validation images
        start_index: Token start index
        im_per_mode: Number of images per mode to display
        dpi: DPI for output figure
        logger: Optional logger for logging information

    Returns:
        Path to the saved figure
    """
    # Initialize logger if not provided
    if logger is None:
        logger = logging.getLogger(__name__)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    if "/" in file_name:
        os.makedirs(os.path.join(output_dir, file_name.split("/")[0]), exist_ok=True)

    num_mode = len(mode_list)
    fig, axs = plt.subplots(
        num_mode,
        im_per_mode * 2 + 1,
        figsize=(im_per_mode * 2 + 1, num_mode * 1.05),
        layout="constrained",
    )

    # Handle case where mode_list has only one item
    if num_mode == 1:
        axs = [axs]

    device = next(hook_model.parameters()).device
    output_path = ""

    try:
        # Process each mode
        for row, (layer, head, mode, sign) in enumerate(mode_list):
            logger.info(f"Processing L{layer} H{head} M{mode} {sign}")

            # Validate indices
            if (
                layer >= U_total.shape[0]
                or head >= U_total.shape[1]
                or mode >= U_total.shape[3]
            ):
                logger.error(
                    f"Invalid indices: L{layer} H{head} M{mode}. Shape: {U_total.shape}"
                )
                continue

            # Get activations and sort
            if sign == "p":
                # Use the pre-computed mode-specific activations
                activations = activations_p[layer, head, mode]
                sorted_indices = np.argsort(activations)
                top_indices = sorted_indices[
                    -im_per_mode:
                ]  # Get highest activation indices

                # Get U and V vectors
                U = torch.from_numpy(U_total[layer, head, :, mode]).float().to(device)
                V = (
                    torch.from_numpy(Vt_total[layer, head, mode, :])
                    .float()
                    .to(device)
                    .T
                )
            else:
                # Use the pre-computed mode-specific activations
                activations = activations_n[layer, head, mode]
                sorted_indices = np.argsort(activations)
                top_indices = sorted_indices[
                    -im_per_mode:
                ]  # Get highest activation indices

                # Get U and V vectors (negative sign)
                U = torch.from_numpy(-U_total[layer, head, :, mode]).float().to(device)
                V = (
                    torch.from_numpy(-Vt_total[layer, head, mode, :])
                    .float()
                    .to(device)
                    .T
                )

            # Display mode info
            dot_product = torch.dot(U, V).detach().cpu().numpy()
            axs[row][0].text(
                0.1,
                0.5,
                f"L{layer} H{head} M{mode} {sign}\n{dot_product:.3f}",
                fontsize=11,
            )
            axs[row][0].set_axis_off()
            axs[row][0].set_xlim((0, 1))

            # Process top images
            for i, idx in enumerate(reversed(top_indices)):
                img_idx = int(idx)
                try:
                    # Load and display original image
                    image_path = os.path.join(imagenet_val_path, im_list[img_idx])
                    im = Image.open(image_path).convert("RGB")
                    im = im.resize((224, 224))
                    axs[row][i * 2 + 1].imshow(np.array(im))
                    axs[row][i * 2 + 1].set_axis_off()

                    # Process image for visualization
                    input_tensor = processor(images=im, return_tensors="pt")
                    input_tensor = {
                        k: v.to(device)
                        for k, v in input_tensor.items()
                        if isinstance(v, torch.Tensor)
                    }

                    # Get hidden states
                    _ = hook_model(
                        input_tensor["pixel_values"],
                        output_hidden_states=False,
                        output_attentions=False,
                    )
                    output = {"hidden_states": hook_model.outputs}
                    hidden_states = output["hidden_states"][layer]
                    hook_model.clear_outputs()

                    # Calculate U and V maps
                    Umap = (
                        (hidden_states[:, :, :] @ U)[0, start_index:]
                        .unflatten(0, (14, 14))
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    Umap_min = np.percentile(Umap, 30)
                    Umap_max = np.percentile(Umap, 95)
                    if Umap_max > Umap_min:
                        Umap = (Umap - Umap_min) / (Umap_max - Umap_min)
                    Umap = np.clip(Umap, 0, 1)

                    Vmap = (
                        (hidden_states[:, :, :] @ V)[0, start_index:]
                        .unflatten(0, (14, 14))
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    Vmap_min = np.percentile(Vmap, 30)
                    Vmap_max = np.percentile(Vmap, 95)
                    if Vmap_max > Vmap_min:
                        Vmap = (Vmap - Vmap_min) / (Vmap_max - Vmap_min)
                    Vmap = np.clip(Vmap, 0, 1)

                    # Resize properly using PIL
                    Umap_img = Image.fromarray((Umap * 255).astype(np.uint8))
                    Vmap_img = Image.fromarray((Vmap * 255).astype(np.uint8))

                    Umap_resized = (
                        np.array(Umap_img.resize((224, 224), Image.BICUBIC)) / 255.0
                    )
                    Vmap_resized = (
                        np.array(Vmap_img.resize((224, 224), Image.BICUBIC)) / 255.0
                    )

                    # Create overlay
                    ratio = 0.6
                    im_array = np.array(im)
                    RGBmap = np.clip(
                        np.stack([Umap_resized, Vmap_resized, Vmap_resized], axis=-1),
                        0,
                        1,
                    ) * ratio + im_array / 255 * (1 - ratio)
                    axs[row][i * 2 + 2].imshow(RGBmap)
                    axs[row][i * 2 + 2].set_axis_off()

                except Exception as e:
                    logger.error(f"Error processing image {img_idx}: {e}")
                    # Show error placeholder in the display
                    axs[row][i * 2 + 1].text(0.1, 0.5, f"Error: {e}", fontsize=8)
                    axs[row][i * 2 + 1].set_axis_off()
                    axs[row][i * 2 + 2].text(0.1, 0.5, f"Error: {e}", fontsize=8)
                    axs[row][i * 2 + 2].set_axis_off()

        # Save the figure
        output_path = os.path.join(output_dir, f"{file_name}.png")
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved visualization to {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error in plot_mode_heatmap: {e}")
        return ""
    finally:
        # Always close the figure to free memory
        plt.close(fig)


def plot_cosine_similarity(
    model_names: List[str],
    data_path: str,
    figure_path: str,
    weighted: bool = False,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Plot the averaged cosine similarity between U and V matrices.

    Args:
        model_names: List of model names to include in the plot
        data_path: Path to the data directory containing UVS files
        figure_path: Path to save the output figure
        weighted: Whether to weight cosine similarity by singular values
        logger: Optional logger for logging information

    Returns:
        Path to the saved figure
    """
    # Initialize logger if not provided
    if logger is None:
        logger = logging.getLogger(__name__)

    # Model style configurations - add more as needed
    model_styles = {
        "vit-base-patch16-224": ("-", "b"),
        "vit-base-patch32-384": ("--", "b"),
        "vit-large-patch16-224": ("-.", "b"),
        "vit-large-patch32-384": (":", "b"),
        "vit-huge-patch14-224-in21k": ("o-", "b"),
        "dino-vits16": ("-", "r"),
        "dino-vitb16": ("--", "r"),
        "clip-vit-base-patch16": ("-", "c"),
        "clip-vit-base-patch32": ("--", "c"),
        "clip-vit-large-patch14": ("-.", "c"),
        "deit-tiny-distilled-patch16-224": ("-", "m"),
        "deit-small-distilled-patch16-224": ("--", "m"),
        "deit-base-distilled-patch16-224": ("-.", "m"),
    }

    # Create output directory
    os.makedirs(os.path.join(figure_path, "cosine"), exist_ok=True)

    # Create figure
    f = plt.figure(figsize=(4.5, 4.5), dpi=300)
    output_path = ""

    try:
        # Process each model
        for model_name in model_names:
            short_name = model_name.split("/")[1]
            linestyle, color = model_styles.get(
                short_name, ("-", "k")
            )  # Default to black solid line

            try:
                # Load U, V, and S matrices
                U_file = os.path.join(data_path, "UVS", f"{model_name}_U_total.npy")
                Vt_file = os.path.join(data_path, "UVS", f"{model_name}_Vt_total.npy")
                S_file = os.path.join(data_path, "UVS", f"{model_name}_S_total.npy")

                logger.info(f"Loading data for {model_name}...")
                U_total = np.load(U_file)
                Vt_total = np.load(Vt_file)
                S_total = np.load(S_file)

                # Calculate cosine similarity
                Ut_total = np.transpose(U_total, [0, 1, 3, 2])
                cosine_similarity = np.einsum("...i,...i", Ut_total, Vt_total)

                if weighted:
                    # Weight by singular values
                    S_total = S_total / np.sum(S_total, axis=2, keepdims=True)
                    cosine_similarity = np.multiply(cosine_similarity, S_total)

                # Average across modes and heads
                cosine_similarity = np.mean(cosine_similarity, axis=2)
                cosine_similarity = np.mean(cosine_similarity, axis=1)

                # Plot
                x_values = [
                    x / (cosine_similarity.shape[0] - 1)
                    for x in range(cosine_similarity.shape[0])
                ]
                plt.plot(
                    x_values,
                    cosine_similarity,
                    color + linestyle,
                    markersize=2,
                    label=short_name,
                )

            except Exception as e:
                logger.error(f"Error processing {model_name}: {e}")

        # Configure plot
        plt.xticks([], [])
        plt.xlabel("Layer")

        if weighted:
            plt.ylabel("Weighted cosine similarity")
            output_path = os.path.join(
                figure_path, "cosine", "average_weighted_cosine.png"
            )
        else:
            plt.ylabel("Cosine similarity")
            output_path = os.path.join(figure_path, "cosine", "average_cosine.png")

        plt.legend(loc="upper right", bbox_to_anchor=(0.8, 1), fontsize=6)

        # Save figure
        f.savefig(output_path, dpi=600, bbox_inches="tight")
        logger.info(f"Saved cosine similarity plot to {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error in plot_cosine_similarity: {e}")
        return ""
    finally:
        plt.close(f)


def plot_spectrum_and_cosine(
    model_name: str,
    data_path: str,
    figure_path: str,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Plot the spectrum and cosine similarity for each layer and head.

    Args:
        model_name: Model name to analyze
        data_path: Path to the data directory containing UVS files
        figure_path: Path to save the output figure
        logger: Optional logger for logging information

    Returns:
        Path to the saved figure
    """
    # Initialize logger if not provided
    if logger is None:
        logger = logging.getLogger(__name__)

    f = None
    output_path = ""

    try:
        logger.info(f"Loading data for {model_name}...")

        # Load matrices
        U_total = np.load(os.path.join(data_path, "UVS", f"{model_name}_U_total.npy"))
        Vt_total = np.load(os.path.join(data_path, "UVS", f"{model_name}_Vt_total.npy"))
        S_total = np.load(os.path.join(data_path, "UVS", f"{model_name}_S_total.npy"))

        Ut_total = np.transpose(U_total, [0, 1, 3, 2])
        cosine_similarity = np.einsum("...i,...i", Ut_total, Vt_total)

        # Calculate baseline for comparison
        dim = Ut_total.shape[3]
        a1 = np.random.rand(10000, dim)
        a2 = np.random.rand(10000, dim)
        a1 = a1 / np.linalg.norm(a1, axis=1, keepdims=True)
        a2 = a2 / np.linalg.norm(a2, axis=1, keepdims=True)
        cosine_similarity_control = np.einsum("...i,...i", a1, a2)
        baseline = np.percentile(cosine_similarity_control, 99)

        n_layer = cosine_similarity.shape[0]
        n_head = cosine_similarity.shape[1]

        # Create figure
        f, axs = plt.subplots(
            n_layer, n_head, figsize=(n_head * 4, n_layer * 3), dpi=100
        )
        f.tight_layout()

        # Plot for each layer and head
        for layer in range(n_layer):
            singular_value_max = np.max(S_total[layer, :, :])
            for head in range(n_head):
                ax1 = axs[layer][head]
                ax1.plot(cosine_similarity[layer, head, :], ".", color="firebrick")
                ax1.set_ylim([-1, 1])
                ax1.set_xlabel("Mode")
                ax1.set_ylabel("Cosine similarity")
                ax1.axhline(y=baseline, color="gray", linestyle="--")
                ax1.axhline(y=-baseline, color="gray", linestyle="--")

                ax2 = ax1.twinx()
                ax2.set_ylim([0, 1.1 * singular_value_max])
                ax2.plot(S_total[layer, head, :], "x", color="royalblue")
                ax2.set_ylabel("Singular value")
                ax2.set_title(f"L{layer}, h{head}")

                f.tight_layout()

        # Save figure
        os.makedirs(os.path.join(figure_path, "cosine"), exist_ok=True)
        output_path = os.path.join(
            figure_path, "cosine", f"{model_name}_cosine_spectrum.png"
        )
        f.savefig(output_path, dpi=100, bbox_inches="tight")
        logger.info(f"Saved spectrum plot to {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error in plot_spectrum_and_cosine: {e}")
        return ""
    finally:
        # Always close the figure to free memory
        if f is not None:
            plt.close(f)
