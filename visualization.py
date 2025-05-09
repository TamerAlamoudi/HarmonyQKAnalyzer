# utils/visualization.py
"""
Visualization utilities for HarmonyQKAnalyzer.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import torch
import logging
from typing import List, Tuple, Optional, Any, Callable
from model_loader import get_model_name_for_files

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
    save_diagnostics: bool = False
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
        save_diagnostics: Whether to save additional diagnostic information
        
    Returns:
        Path to the saved figure
    """
    # Initialize logger if not provided
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    if '/' in file_name:
        os.makedirs(os.path.join(output_dir, file_name.split('/')[0]), exist_ok=True)
    
    # Create a diagnostics directory if needed
    diagnostics_dir = os.path.join(output_dir, "diagnostics")
    if save_diagnostics:
        os.makedirs(diagnostics_dir, exist_ok=True)
    
    num_mode = len(mode_list)
    fig, axs = plt.subplots(num_mode, im_per_mode*2+1, figsize=(im_per_mode*2+1, num_mode*1.25), layout="constrained")
    
    # Handle case where mode_list has only one item
    if num_mode == 1:
        axs = [axs]
    
    device = next(hook_model.parameters()).device
    output_path = ""
    
    # Create a small legend figure for the heatmap colors
    legend_fig, legend_ax = plt.subplots(figsize=(5, 1))
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    
    # Create a color map legend
    legend_ax.imshow(gradient, aspect='auto', cmap=plt.cm.jet)
    legend_ax.text(0, 0, "Low", ha='left', va='center', color='white', fontsize=8)
    legend_ax.text(255, 0, "High", ha='right', va='center', color='white', fontsize=8)
    legend_ax.set_title('Activation Strength', fontsize=10)
    legend_ax.set_yticks([])
    legend_ax.set_xticks([])
    
    # Add color coding explanation
    legend_ax.text(85, 0, "Red: U activation", ha='center', va='center', color='white', fontsize=8)
    legend_ax.text(170, 0, "Blue: V activation", ha='center', va='center', color='white', fontsize=8)
    legend_ax.text(128, 0, "Purple: Both", ha='center', va='center', color='white', fontsize=8)
    
    legend_path = os.path.join(output_dir, "heatmap_legend.png")
    legend_fig.savefig(legend_path, dpi=150, bbox_inches='tight')
    plt.close(legend_fig)
    
    try:
        # Print summary of processing task
        logger.info(f"Processing {num_mode} modes with {im_per_mode} images per mode")
        logger.info(f"Total visualizations to create: {num_mode * im_per_mode}")
        
        # Process each mode
        for row, (layer, head, mode, sign) in enumerate(mode_list):
            logger.info(f"Processing L{layer} H{head} M{mode} {sign}")
            
            # Validate indices
            if layer >= U_total.shape[0] or head >= U_total.shape[1] or mode >= U_total.shape[3]:
                logger.error(f"Invalid indices: L{layer} H{head} M{mode}. Shape: {U_total.shape}")
                continue
            
            # Get activations and sort
            if sign == "p":
                # Use the pre-computed mode-specific activations
                activations = activations_p[layer, head, mode]
                sorted_indices = np.argsort(activations)
                top_indices = sorted_indices[-im_per_mode:]  # Get highest activation indices
                
                # Get U and V vectors
                U = torch.from_numpy(U_total[layer, head, :, mode]).float().to(device)
                V = torch.from_numpy(Vt_total[layer, head, mode, :]).float().to(device).T
            else:
                # Use the pre-computed mode-specific activations
                activations = activations_n[layer, head, mode]
                sorted_indices = np.argsort(activations)
                top_indices = sorted_indices[-im_per_mode:]  # Get highest activation indices
                
                # Get U and V vectors (negative sign)
                U = torch.from_numpy(-U_total[layer, head, :, mode]).float().to(device)
                V = torch.from_numpy(-Vt_total[layer, head, mode, :]).float().to(device).T
            
            # Calculate and display comprehensive mode info
            dot_product = torch.dot(U, V).detach().cpu().numpy()
            
            # Calculate additional stats for the mode
            norm_u = torch.norm(U).detach().cpu().numpy()
            norm_v = torch.norm(V).detach().cpu().numpy()
            cosine_sim = dot_product / (norm_u * norm_v)
            
            # Display more detailed info in the first column
            info_text = (f"L{layer} H{head} M{mode} {sign}\n"
                       f"Dot: {dot_product:.3f}\n"
                       f"Cos: {cosine_sim:.3f}")
            axs[row][0].text(0.1, 0.5, info_text, fontsize=9)
            axs[row][0].set_axis_off()
            axs[row][0].set_xlim((0,1))
            
            # Calculate activation statistics for diagnostics
            activation_mean = np.mean(activations)
            activation_std = np.std(activations)
            activation_min = np.min(activations)
            activation_max = np.max(activations)
            
            # If save diagnostic information
            if save_diagnostics:
                mode_dir = os.path.join(diagnostics_dir, f"L{layer}_H{head}_M{mode}_{sign}")
                os.makedirs(mode_dir, exist_ok=True)
                
                # Save activation histogram
                hist_fig, hist_ax = plt.subplots(figsize=(8, 6))
                hist_ax.hist(activations, bins=50)
                hist_ax.set_title(f"Activation Distribution L{layer} H{head} M{mode} {sign}")
                hist_ax.set_xlabel("Activation Value")
                hist_ax.set_ylabel("Frequency")
                hist_ax.axvline(x=activations[top_indices[-1]], color='r', linestyle='--', 
                              label=f'Min Selected ({activations[top_indices[-1]]:.3f})')
                hist_ax.legend()
                hist_fig.savefig(os.path.join(mode_dir, "activation_histogram.png"), dpi=150)
                plt.close(hist_fig)
                
                # Save U and V vector visualizations
                u_np = U.detach().cpu().numpy()
                v_np = V.detach().cpu().numpy()
                
                # Create vector plots
                vec_fig, (u_ax, v_ax) = plt.subplots(1, 2, figsize=(12, 5))
                u_ax.stem(u_np)
                u_ax.set_title(f"U Vector (L{layer} H{head} M{mode})")
                v_ax.stem(v_np)
                v_ax.set_title(f"V Vector (L{layer} H{head} M{mode})")
                vec_fig.savefig(os.path.join(mode_dir, "vectors.png"), dpi=150)
                plt.close(vec_fig)
            
            # Process top images
            logger.info(f"  Processing {im_per_mode} images for L{layer} H{head} M{mode} {sign}")
            all_combined_maps = []
            
            for i, idx in enumerate(reversed(top_indices)):
                img_idx = int(idx)
                if i % 4 == 0:  # Log every few images to keep output manageable
                    logger.info(f"    Processing image {i+1}/{im_per_mode} (index: {img_idx})")
                try:
                    # Load and display original image
                    image_path = os.path.join(imagenet_val_path, im_list[img_idx])
                    im = Image.open(image_path).convert('RGB')
                    im = im.resize((224,224))
                    axs[row][i*2+1].imshow(np.array(im))
                    axs[row][i*2+1].set_axis_off()
                    
                    # Add activation value as text on the image
                    act_value = activations[img_idx]
                    axs[row][i*2+1].text(5, 20, f"{act_value:.3f}", fontsize=8, color='white', 
                                      bbox=dict(facecolor='black', alpha=0.5))
                    
                    # Process image for visualization
                    input_tensor = processor(images=im, return_tensors="pt")
                    input_tensor = {k: v.to(device) for k, v in input_tensor.items() if isinstance(v, torch.Tensor)}
                    
                    # Get hidden states
                    _ = hook_model(input_tensor["pixel_values"], output_hidden_states=False, output_attentions=False)
                    output = {'hidden_states': hook_model.outputs}
                    hidden_states = output['hidden_states'][layer]
                    hook_model.clear_outputs()
                    
                    # Calculate U and V maps
                    Umap = (hidden_states[:,:,:] @ U)[0, start_index:].unflatten(0,(14,14)).detach().cpu().numpy()
                    # Use 10th-90th percentile for better contrast
                    Umap_min = np.percentile(Umap, 10)
                    Umap_max = np.percentile(Umap, 90)
                    if Umap_max > Umap_min:
                        Umap = (Umap - Umap_min)/(Umap_max - Umap_min)
                    Umap = np.clip(Umap, 0, 1)
                    
                    Vmap = (hidden_states[:,:,:] @ V)[0, start_index:].unflatten(0,(14,14)).detach().cpu().numpy()
                    # Use 10th-90th percentile for better contrast
                    Vmap_min = np.percentile(Vmap, 10)
                    Vmap_max = np.percentile(Vmap, 90)
                    if Vmap_max > Vmap_min:
                        Vmap = (Vmap - Vmap_min)/(Vmap_max - Vmap_min)
                    Vmap = np.clip(Vmap, 0, 1)
                    
                    # Create a more detailed combined activation map (U*V)
                    combined_map = Umap * Vmap
                    combined_min = np.percentile(combined_map, 10)
                    combined_max = np.percentile(combined_map, 90)
                    if combined_max > combined_min:
                        combined_map = (combined_map - combined_min)/(combined_max - combined_min)
                    combined_map = np.clip(combined_map, 0, 1)
                    
                    # Store the combined map for later analysis
                    all_combined_maps.append(combined_map)
                    
                    # Resize properly using PIL - use LANCZOS for better quality
                    Umap_img = Image.fromarray((Umap * 255).astype(np.uint8))
                    Vmap_img = Image.fromarray((Vmap * 255).astype(np.uint8))
                    combined_img = Image.fromarray((combined_map * 255).astype(np.uint8))
                    
                    Umap_resized = np.array(Umap_img.resize((224, 224), Image.LANCZOS)) / 255.0
                    Vmap_resized = np.array(Vmap_img.resize((224, 224), Image.LANCZOS)) / 255.0
                    combined_resized = np.array(combined_img.resize((224, 224), Image.LANCZOS)) / 255.0
                    
                    # Create overlay with better color mapping:
                    # Red channel: U activation
                    # Blue channel: V activation
                    # Green channel: combined activation
                    ratio = 0.6  # Opacity of overlay
                    im_array = np.array(im) / 255.0
                    
                    # Create RGB heatmap with distinct colors for U and V
                    heatmap = np.zeros((224, 224, 3))
                    heatmap[:, :, 0] = Umap_resized  # Red channel for U
                    heatmap[:, :, 1] = combined_resized  # Green channel for combined
                    heatmap[:, :, 2] = Vmap_resized  # Blue channel for V
                    
                    # Apply overlay
                    overlay = heatmap * ratio + im_array * (1 - ratio)
                    overlay = np.clip(overlay, 0, 1)
                    
                    axs[row][i*2+2].imshow(overlay)
                    axs[row][i*2+2].set_axis_off()
                    
                    # Save individual heatmaps for diagnostics if requested
                    if save_diagnostics:
                        img_dir = os.path.join(mode_dir, f"image_{img_idx}")
                        os.makedirs(img_dir, exist_ok=True)
                        
                        # Save original image
                        im.save(os.path.join(img_dir, "original.jpg"))
                        
                        # Save heatmaps
                        plt.imsave(os.path.join(img_dir, "u_heatmap.png"), Umap_resized, cmap='jet')
                        plt.imsave(os.path.join(img_dir, "v_heatmap.png"), Vmap_resized, cmap='jet')
                        plt.imsave(os.path.join(img_dir, "combined_heatmap.png"), combined_resized, cmap='jet')
                        
                        # Save the overlay
                        plt.imsave(os.path.join(img_dir, "overlay.png"), overlay)
                    
                except Exception as e:
                    logger.error(f"Error processing image {img_idx}: {e}")
                    # Show error placeholder in the display
                    axs[row][i*2+1].text(0.1, 0.5, f"Error: {e}", fontsize=8)
                    axs[row][i*2+1].set_axis_off()
                    axs[row][i*2+2].text(0.1, 0.5, f"Error: {e}", fontsize=8)
                    axs[row][i*2+2].set_axis_off()
            
            # Analyze consistency across images if we have diagnostics enabled
            if save_diagnostics and len(all_combined_maps) > 0:
                # Convert to same size arrays
                sized_maps = [np.array(Image.fromarray((m * 255).astype(np.uint8)).resize((56, 56), Image.LANCZOS))/255.0
                           for m in all_combined_maps]
                
                # Calculate average activation map
                avg_map = np.mean(sized_maps, axis=0)
                
                # Save average map
                plt.figure(figsize=(6, 6))
                plt.imshow(avg_map, cmap='jet')
                plt.title(f"Average Activation Pattern L{layer} H{head} M{mode} {sign}")
                plt.colorbar()
                plt.savefig(os.path.join(mode_dir, "average_activation.png"), dpi=150)
                plt.close()
        
        # Save the figure
        output_path = os.path.join(output_dir, f"{file_name}.png")
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
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
    weights_paths: Optional[List[str]] = None,
    file_prefixes: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None
) -> str:
    """
    Plot the averaged cosine similarity between U and V matrices.
    
    Args:
        model_names: List of model names to include in the plot
        data_path: Path to the data directory containing UVS files
        figure_path: Path to save the output figure
        weighted: Whether to weight cosine similarity by singular values
        weights_paths: List of paths to custom weights files (optional, same length as model_names)
        file_prefixes: List of file prefixes to use for consistency (optional, same length as model_names)
        logger: Optional logger for logging information
        
    Returns:
        Path to the saved figure
    """
    # Initialize logger if not provided
    if logger is None:
        logger = logging.getLogger(__name__)
    
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
    
    # Model style configurations - add more as needed
    model_styles = {
        "vit-base-patch16-224": ('-', 'b'),
        "vit-base-patch32-384": ('--', 'b'),
        "vit-large-patch16-224": ('-.', 'b'),
        "vit-large-patch32-384": (':', 'b'),
        "vit-huge-patch14-224-in21k": ('o-', 'b'),
        "dino-vits16": ('-', 'r'),
        "dino-vitb16": ('--', 'r'),
        "clip-vit-base-patch16": ('-', 'c'),
        "clip-vit-base-patch32": ('--', 'c'),
        "clip-vit-large-patch14": ('-.', 'c'),
        "deit-tiny-distilled-patch16-224": ('-', 'm'),
        "deit-small-distilled-patch16-224": ('--', 'm'),
        "deit-base-distilled-patch16-224": ('-.', 'm')
    }
    
    # Create output directory
    os.makedirs(os.path.join(figure_path, "cosine"), exist_ok=True)
    
    # Create figure
    f = plt.figure(figsize=(8, 6), dpi=300)
    output_path = ""
    
    # Define a custom file name suffix based on the models being compared
    custom_suffix = ""
    if len(model_names) <= 3:  # Only create custom names for small comparisons
        model_key_parts = []
        for i, model_name in enumerate(model_names):
            prefix = file_prefixes[i]
            if prefix:
                # Extract meaningful parts from prefixes like 'vit_b16_canonical' -> 'canonical'
                if 'canonical' in prefix:
                    model_key_parts.append('canonical')
                elif 'harmonized' in prefix:
                    model_key_parts.append('harmonized')
                else:
                    model_key_parts.append(prefix.split('_')[-1])  # Use last part of prefix
            else:
                short_name = model_name.split('/')[1] if '/' in model_name else model_name
                model_key_parts.append(short_name.split('-')[0])  # Just use first part
                
        custom_suffix = '_'.join(model_key_parts)
        if custom_suffix:
            custom_suffix = '_' + custom_suffix
    
    try:
        # Process each model
        for i, model_name in enumerate(model_names):
            short_name = model_name.split('/')[1] if '/' in model_name else model_name
            linestyle, color = model_styles.get(short_name, ('-', 'k'))  # Default to black solid line
            
            try:
                # Load U, V, and S matrices
                weights_path = weights_paths[i]
                file_prefix = file_prefixes[i]
                safe_model_name = get_model_name_for_files(model_name, weights_path, file_prefix)
                U_file = os.path.join(data_path, "UVS", f"{safe_model_name}_U_total.npy")
                Vt_file = os.path.join(data_path, "UVS", f"{safe_model_name}_Vt_total.npy")
                S_file = os.path.join(data_path, "UVS", f"{safe_model_name}_S_total.npy")
                
                logger.info(f"Loading data for {model_name} (file prefix: {safe_model_name})...")
                U_total = np.load(U_file)
                Vt_total = np.load(Vt_file)
                S_total = np.load(S_file)
                
                # Calculate cosine similarity
                Ut_total = np.transpose(U_total, [0, 1, 3, 2])
                cosine_similarity = np.einsum('...i,...i', Ut_total, Vt_total)
                
                if weighted:
                    # Weight by singular values
                    S_total = S_total / np.sum(S_total, axis=2, keepdims=True)
                    cosine_similarity = np.multiply(cosine_similarity, S_total)
                
                # Average across modes and heads
                cosine_similarity = np.mean(cosine_similarity, axis=2)
                cosine_similarity = np.mean(cosine_similarity, axis=1)
                
                # Create a more descriptive label using file prefix if available
                if file_prefix:
                    # Create more descriptive label based on file prefix
                    if 'canonical' in file_prefix:
                        display_name = f"Canonical {short_name}"
                        # Use blue for canonical models
                        color = 'b'
                    elif 'harmonized' in file_prefix:
                        display_name = f"Harmonized {short_name}"
                        # Use red for harmonized models
                        color = 'r'
                    else:
                        display_name = f"{file_prefix} {short_name}"
                else:
                    display_name = short_name
                
                # Plot
                x_values = [x / (cosine_similarity.shape[0] - 1) for x in range(cosine_similarity.shape[0])]
                plt.plot(x_values, cosine_similarity, color + linestyle, markersize=4, label=display_name, linewidth=2.5)
                
            except Exception as e:
                logger.error(f"Error processing {model_name}: {e}")
        
        # Configure plot
        plt.xticks(x_values, [str(i) for i in range(len(x_values))], fontsize=10)
        plt.xlabel('Layer', fontsize=12)
        
        if weighted:
            plt.ylabel('Weighted cosine similarity', fontsize=12)
            output_path = os.path.join(figure_path, "cosine", f"average_weighted_cosine{custom_suffix}.png")
        else:
            plt.ylabel('Cosine similarity', fontsize=12)
            output_path = os.path.join(figure_path, "cosine", f"average_cosine{custom_suffix}.png")
        
        # Add a grid for better readability
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Better legend placement and formatting
        plt.legend(loc='best', fontsize=10, framealpha=0.9)
        
        # Add a light gray box around the plot
        plt.box(True)
        
        # Add a title if comparing specific models
        if len(model_names) == 2 and any('canonical' in p for p in file_prefixes) and any('harmonized' in p for p in file_prefixes):
            plt.title('Comparison of Canonical vs. Harmonized Model', fontsize=14)
        
        # Save figure
        f.savefig(output_path, dpi=600, bbox_inches='tight')
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
    weights_path: Optional[str] = None,
    file_prefix: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> str:
    """
    Plot the spectrum and cosine similarity for each layer and head.
    
    Args:
        model_name: Model name to analyze
        data_path: Path to the data directory containing UVS files
        figure_path: Path to save the output figure
        weights_path: Path to custom weights file (optional)
        file_prefix: File prefix to use for consistency (optional)
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
        safe_model_name = get_model_name_for_files(model_name, weights_path, file_prefix)
        logger.info(f"Using file prefix: {safe_model_name}")
        
        U_path = os.path.join(data_path, "UVS", f"{safe_model_name}_U_total.npy")
        Vt_path = os.path.join(data_path, "UVS", f"{safe_model_name}_Vt_total.npy")
        S_path = os.path.join(data_path, "UVS", f"{safe_model_name}_S_total.npy")
        
        logger.info(f"Loading U matrix from {U_path}")
        U_total = np.load(U_path)
        
        logger.info(f"Loading Vt matrix from {Vt_path}")
        Vt_total = np.load(Vt_path)
        
        logger.info(f"Loading S matrix from {S_path}")
        S_total = np.load(S_path)
        
        Ut_total = np.transpose(U_total, [0, 1, 3, 2])
        cosine_similarity = np.einsum('...i,...i', Ut_total, Vt_total)
        
        # Calculate baseline for comparison
        dim = Ut_total.shape[3]
        a1 = np.random.rand(10000, dim)
        a2 = np.random.rand(10000, dim)
        a1 = a1 / np.linalg.norm(a1, axis=1, keepdims=True)
        a2 = a2 / np.linalg.norm(a2, axis=1, keepdims=True)
        cosine_similarity_control = np.einsum('...i,...i', a1, a2)
        baseline = np.percentile(cosine_similarity_control, 99)
        
        n_layer = cosine_similarity.shape[0]
        n_head = cosine_similarity.shape[1]
        
        # Create figure
        f, axs = plt.subplots(n_layer, n_head, figsize=(n_head*4, n_layer*3), dpi=100)
        f.tight_layout()
        
        # Plot for each layer and head
        for layer in range(n_layer):
            singular_value_max = np.max(S_total[layer, :, :])
            for head in range(n_head):
                ax1 = axs[layer][head]
                ax1.plot(cosine_similarity[layer, head, :], ".", color='firebrick')
                ax1.set_ylim([-1, 1])
                ax1.set_xlabel('Mode')
                ax1.set_ylabel('Cosine similarity')
                ax1.axhline(y=baseline, color='gray', linestyle='--')
                ax1.axhline(y=-baseline, color='gray', linestyle='--')
                
                ax2 = ax1.twinx()
                ax2.set_ylim([0, 1.1*singular_value_max])
                ax2.plot(S_total[layer, head, :], "x", color='royalblue')
                ax2.set_ylabel('Singular value')
                ax2.set_title(f"L{layer}, h{head}")
                
                f.tight_layout()
        
        # Save figure
        os.makedirs(os.path.join(figure_path, "cosine"), exist_ok=True)
        safe_model_name = get_model_name_for_files(model_name, weights_path, file_prefix)
        output_path = os.path.join(figure_path, "cosine", f"{safe_model_name}_cosine_spectrum.png")
        f.savefig(output_path, dpi=100, bbox_inches='tight')
        logger.info(f"Saved spectrum plot to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error in plot_spectrum_and_cosine: {e}")
        return ""
    finally:
        # free memory
        if f is not None:
            plt.close(f)