# models/model_analysis.py
"""
Tools for analyzing vision transformer models.
"""
import numpy as np
import os
import multiprocessing
import time
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Any, Union, Callable

class LayerNormHook(torch.nn.Module):
    """Hook model to capture layer norm outputs for ViT models."""
    
    def __init__(self, model: torch.nn.Module):
        """
        Initialize hook model.
        
        Args:
            model: Base model to hook into
        """
        super().__init__()
        self.model = model
        self.outputs = []
        self.hooks = []

        for layer in self.model.vit.encoder.layer:
            hook = layer.layernorm_before.register_forward_hook(self.hook_fn)
            self.hooks.append(hook)

    def hook_fn(self, module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
        """Hook function to capture outputs."""
        self.outputs.append(output.detach())
        
    def clear_outputs(self) -> None:
        """Clear captured outputs."""
        self.outputs = []
        import gc
        gc.collect()

    def remove_hooks(self) -> None:
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
    
    def forward(self, input: torch.Tensor, output_hidden_states: bool = False, output_attentions: bool = False) -> Any:
        """Forward pass through the model."""
        with torch.no_grad():
            output = self.model(input, output_hidden_states=output_hidden_states, output_attentions=output_attentions)
        return output

class LayerNormHook_deit(torch.nn.Module):
    """Hook model to capture layer norm outputs for DeiT models."""
    
    def __init__(self, model: torch.nn.Module):
        """
        Initialize hook model.
        
        Args:
            model: Base model to hook into
        """
        super().__init__()
        self.model = model
        self.outputs = []
        self.hooks = []

        for layer in self.model.deit.encoder.layer:
            hook = layer.layernorm_before.register_forward_hook(self.hook_fn)
            self.hooks.append(hook)

    def hook_fn(self, module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
        """Hook function to capture outputs."""
        self.outputs.append(output.detach())
        
    def clear_outputs(self) -> None:
        """Clear captured outputs."""
        self.outputs = []
        import gc
        gc.collect()

    def remove_hooks(self) -> None:
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
    
    def forward(self, input: torch.Tensor, output_hidden_states: bool = False, output_attentions: bool = False) -> Any:
        """Forward pass through the model."""
        with torch.no_grad():
            output = self.model(input, output_hidden_states=output_hidden_states, output_attentions=output_attentions)
        return output

class LayerNormHook_clip(torch.nn.Module):
    """Hook model to capture layer norm outputs for CLIP models."""
    
    def __init__(self, model: torch.nn.Module):
        """
        Initialize hook model.
        
        Args:
            model: Base model to hook into
        """
        super().__init__()
        self.model = model
        self.outputs = []
        self.hooks = []

        for layer in self.model.encoder.layers:
            hook = layer.layer_norm1.register_forward_hook(self.hook_fn)
            self.hooks.append(hook)

    def hook_fn(self, module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
        """Hook function to capture outputs."""
        self.outputs.append(output.detach())
        
    def clear_outputs(self) -> None:
        """Clear captured outputs."""
        self.outputs = []
        import gc
        gc.collect()

    def remove_hooks(self) -> None:
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
    
    def forward(self, input: torch.Tensor, output_hidden_states: bool = False, output_attentions: bool = False) -> Any:
        """Forward pass through the model."""
        with torch.no_grad():
            output = self.model(input, output_hidden_states=output_hidden_states, output_attentions=output_attentions)
        return output

class ModelAnalyzer:
    """Tools for analyzing transformer models."""
    
    def __init__(self, model: torch.nn.Module):
        """
        Initialize analyzer with a model.
        
        Args:
            model: Model to analyze
        """
        self.layer_list = [(name, param.to(torch.device('cpu'))) for name, param in model.named_parameters()]

    def block_diagonalize(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Take skew symmetric matrix and put in QSQ^T.
        
        Args:
            A: Skew symmetric matrix
            
        Returns:
            Tuple containing:
                - Block diagonal matrix
                - Eigenvectors
                - Imaginary eigenvalues
        """
        # A must be a antisymmetric (or skew symmetric) matrix
        assert np.array_equal(A, -A.T)

        d = A.shape[0]

        eigenvalues, eigenvectors = np.linalg.eig(A)
        imageigenvalues = np.imag(eigenvalues)
        realeigenvectors = np.real(eigenvectors)
        imageigenvectors = np.imag(eigenvectors)

        block_diag = np.zeros((d, d))
        V = []

        if d % 2 == 0:  # even dimension is an easy case
            for i in range(int(d / 2)):
                block_diag[2 * i, 2 * i + 1] = imageigenvalues[2 * i]
                block_diag[2 * i + 1, 2 * i] = imageigenvalues[2 * i + 1]
                V.append(realeigenvectors[:, 2 * i] * np.sqrt(2))
                V.append(imageigenvectors[:, 2 * i] * np.sqrt(2))
        else:  # if odd dimension, there is a 0 eigen value and vector.
            find0 = 0
            for i in range(int((d - 1) / 2)):
                # need to skip 0 dimension
                if imageigenvalues[2 * i] == 0:
                    block_diag[2 * i, 2 * i] = 0
                    V.append(realeigenvectors[:, 2 * i])
                    find0 = 1
                block_diag[2 * i + find0, 2 * i + 1 + find0] = imageigenvalues[2 * i + find0]
                block_diag[2 * i + 1 + find0, 2 * i + find0] = imageigenvalues[2 * i + 1 + find0]
                V.append(realeigenvectors[:, 2 * i + find0] * np.sqrt(2))
                V.append(imageigenvectors[:, 2 * i + find0] * np.sqrt(2))
            if imageigenvalues[d - 1] == 0:  # check boundary condition
                block_diag[d - 1, d - 1] = 0
                V.append(realeigenvectors[:, d - 1])

        V = np.array(V)
        V = V.T

        return block_diag, V, imageigenvalues
    
    def svd_angles(self, M: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute SVD and angles between left and right singular vectors.
        
        Args:
            M: Matrix to analyze
            
        Returns:
            Tuple containing:
                - U: Left singular vectors
                - S: Singular values
                - Vt: Right singular vectors transposed
                - angles: Angles between matching singular vectors
        """
        U, S, Vt = np.linalg.svd(M)

        # Compute the angles between left and right singular vectors
        angles = []
        for i in range(min(M.shape)):  # assumes 2D
            u = U[:, i]
            v = Vt[i, :]
            angle = np.arccos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
            angles.append(angle)

        return U, S, Vt, np.array(angles)
    
    def get_QK(self, n_heads: int) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        """
        Extract Q and K matrices from the model.
        
        Args:
            n_heads: Number of attention heads
            
        Returns:
            List of (Q, K, layer_name) tuples for each attention layer
        """
        QK_list = []
        Q = None
        K = None
        unformatted_layer_name = None

        for name, param in self.layer_list:
            if ("key" in name or "k_proj" in name) and "weight" in name:
                K = param.T
                K = K.reshape(K.shape[0], n_heads, -1).detach().cpu().numpy()
                unformatted_layer_name = name.split(".")
            elif ("query" in name or "q_proj" in name) and "weight" in name:
                Q = param.T
                Q = Q.reshape(Q.shape[0], n_heads, -1).detach().cpu().numpy()
                
            # Store if we found a key,query pair 
            if K is not None and Q is not None:
                name_list = []
                
                # Store tree up to the 'key' branch
                for term in unformatted_layer_name:
                    if term == "key" or term == "k_proj":
                        break
                    name_list.append(term)
                    
                formatted_layer_name = ("/").join(name_list)
                QK_list.append((np.array(Q), np.array(K), formatted_layer_name))

                # Reset key, query pair for new search
                Q = None
                K = None
                
        return QK_list

def compare_singular_values(
    s_canonical: np.ndarray,
    s_harmonized: np.ndarray,
    layer: int = None,
    head: int = None,
) -> dict:
    """
    Compare singular values between canonical and harmonized models.
    
    Args:
        s_canonical: Singular values for canonical model [layers, heads, modes]
        s_harmonized: Singular values for harmonized model [layers, heads, modes]
        layer: Specific layer to analyze (None for all)
        head: Specific head to analyze (None for all)
        
    Returns:
        Dictionary with comparison metrics
    """
    results = {}
    
    # Handle single layer/head case
    if layer is not None and head is not None:
        s_can = s_canonical[layer, head]
        s_harm = s_harmonized[layer, head]
        
        # Calculate ratios
        ratios = s_harm / np.maximum(s_can, 1e-10)  # Avoid division by zero
        
        # Calculate relative importance shifts
        importance_can = s_can / np.sum(s_can)
        importance_harm = s_harm / np.sum(s_harm)
        importance_shifts = importance_harm - importance_can
        
        # Calculate KL divergence (how distribution changed)
        # Normalize to create proper distributions
        p = importance_can + 1e-10  # small epsilon to avoid log(0)
        q = importance_harm + 1e-10
        p = p / np.sum(p)
        q = q / np.sum(q)
        kl_div = np.sum(p * np.log(p / q))
        
        results = {
            'ratios': ratios,
            'importance_shifts': importance_shifts,
            'kl_divergence': kl_div,
            'top_modes_canonical': np.argsort(-s_can)[:5],
            'top_modes_harmonized': np.argsort(-s_harm)[:5],
            'concentration_canonical': np.sum(s_can[:5]) / np.sum(s_can),
            'concentration_harmonized': np.sum(s_harm[:5]) / np.sum(s_harm)
        }
        return results
    
    # For all layers/heads
    num_layers, num_heads = s_canonical.shape[:2]
    all_results = {}
    
    # Aggregate metrics across all pairs
    kl_divs = np.zeros((num_layers, num_heads))
    concentration_diffs = np.zeros((num_layers, num_heads))
    
    for l in range(num_layers):
        for h in range(num_heads):
            layer_head_results = compare_singular_values(s_canonical, s_harmonized, l, h)
            kl_divs[l, h] = layer_head_results['kl_divergence']
            concentration_diffs[l, h] = (layer_head_results['concentration_harmonized'] - 
                                        layer_head_results['concentration_canonical'])
    
    all_results['kl_divergences'] = kl_divs
    all_results['concentration_diffs'] = concentration_diffs
    all_results['avg_kl_by_layer'] = np.mean(kl_divs, axis=1)
    all_results['avg_concentration_diff_by_layer'] = np.mean(concentration_diffs, axis=1)
    
    return all_results


def compute_vector_similarity(
    u_canonical: np.ndarray,
    v_canonical: np.ndarray,
    u_harmonized: np.ndarray,
    v_harmonized: np.ndarray,
    layer: int = None,
    head: int = None,
    mode: int = None
) -> dict:
    """
    Compute similarity metrics between mode vectors of canonical and harmonized models.
    
    Args:
        u_canonical: U vectors for canonical model
        v_canonical: V vectors for canonical model (transposed)
        u_harmonized: U vectors for harmonized model
        v_harmonized: V vectors for harmonized model (transposed)
        layer: Specific layer to analyze (None for all)
        head: Specific head to analyze (None for all)
        mode: Specific mode to analyze (None for all)
        
    Returns:
        Dictionary with similarity metrics
    """
    results = {}
    
    # Handle single layer/head/mode case
    if layer is not None and head is not None and mode is not None:
        # Extract U vectors
        u_can = u_canonical[layer, head, mode]
        u_harm = u_harmonized[layer, head, mode]
        
        # Extract V vectors - handle different possible shapes
        # Since we're dealing with Vt (transposed), the mode may be in different axis
        if len(v_canonical.shape) == 3:
            # If V is already in [layer, head, mode] format
            v_can = v_canonical[layer, head, mode]
            v_harm = v_harmonized[layer, head, mode]
        elif len(v_canonical.shape) == 4:
            # Handle transposed format [layer, head, features, modes]
            if v_canonical.shape[2] < v_canonical.shape[3]:
                # This is [layer, head, features, modes] format
                v_can = v_canonical[layer, head, :, mode]
                v_harm = v_harmonized[layer, head, :, mode]
            else:
                # This is [layer, head, modes, features] format
                v_can = v_canonical[layer, head, mode]
                v_harm = v_harmonized[layer, head, mode]
        elif len(v_canonical.shape) == 5:
            # If Vt is stored with an extra dimension for transposition
            v_can = v_canonical[layer, head, mode, :]
            v_harm = v_harmonized[layer, head, mode, :]
        else:
            # Handle other possible formats based on visual inspection
            if mode < v_canonical.shape[-1]:
                if layer < v_canonical.shape[0] and head < v_canonical.shape[1]:
                    # Try to access as [layer, head, :, mode]
                    v_can = v_canonical[layer, head, :, mode]
                    v_harm = v_harmonized[layer, head, :, mode]
                else:
                    raise ValueError(f"Cannot extract V vector for layer {layer}, head {head}, mode {mode}")
            else:
                raise ValueError(f"Mode {mode} is out of bounds for V vector with shape {v_canonical.shape}")
        
        # Ensure vectors are 1-dimensional
        u_can = np.ravel(u_can)
        v_can = np.ravel(v_can)
        u_harm = np.ravel(u_harm)
        v_harm = np.ravel(v_harm)
        
        # Normalize vectors for cosine similarity
        u_can_norm = u_can / np.maximum(np.linalg.norm(u_can), 1e-10)
        v_can_norm = v_can / np.maximum(np.linalg.norm(v_can), 1e-10)
        u_harm_norm = u_harm / np.maximum(np.linalg.norm(u_harm), 1e-10)
        v_harm_norm = v_harm / np.maximum(np.linalg.norm(v_harm), 1e-10)
        
        # Compute cosine similarities
        u_similarity = np.abs(np.dot(u_can_norm, u_harm_norm))
        v_similarity = np.abs(np.dot(v_can_norm, v_harm_norm))
        
        # Compute changes in sparsity
        u_sparsity_can = np.sum(np.abs(u_can) < 0.01) / len(u_can)
        u_sparsity_harm = np.sum(np.abs(u_harm) < 0.01) / len(u_harm)
        v_sparsity_can = np.sum(np.abs(v_can) < 0.01) / len(v_can)
        v_sparsity_harm = np.sum(np.abs(v_harm) < 0.01) / len(v_harm)
        
        results = {
            'u_cosine_similarity': u_similarity,
            'v_cosine_similarity': v_similarity,
            'combined_similarity': (u_similarity + v_similarity) / 2,
            'u_sparsity_change': u_sparsity_harm - u_sparsity_can,
            'v_sparsity_change': v_sparsity_harm - v_sparsity_can,
            'alignment_score': 1 - (u_similarity + v_similarity) / 2  # Lower == more aligned
        }
        return results
    
    # For all layers/heads/modes
    # Check the shape and handle appropriately
    if len(u_canonical.shape) == 3:
        num_layers, num_heads, num_modes = u_canonical.shape
    elif len(u_canonical.shape) == 4:
        # Handle case where U is [layers, heads, modes, features]
        num_layers, num_heads, num_modes, _ = u_canonical.shape
    else:
        raise ValueError(f"Unexpected shape for U vectors: {u_canonical.shape}")
    
    # Make sure we don't try to access modes that don't exist
    # Check V shape to determine max modes
    if len(v_canonical.shape) >= 3:
        # For typical V shape [layers, heads, features, modes]
        if len(v_canonical.shape) == 4 and v_canonical.shape[2] < v_canonical.shape[3]:
            # This is the [layers, heads, features, modes] format
            print(f"Detected transposed V format with features={v_canonical.shape[2]}, modes={v_canonical.shape[3]}")
            # Use the minimum number of modes available in both U and V
            num_modes = min(64, num_modes)  # Hard limit to 64 modes which is the feature dimension
        else:
            max_modes_v = min(num_modes, v_canonical.shape[2] if len(v_canonical.shape) == 3 else v_canonical.shape[-1])
            num_modes = min(num_modes, max_modes_v)

    print(f"Analyzing {num_layers} layers, {num_heads} heads, {num_modes} modes")
    print(f"U canonical shape: {u_canonical.shape}, V canonical shape: {v_canonical.shape}")
        
    all_similarities = np.zeros((num_layers, num_heads, num_modes))
    
    for l in range(num_layers):
        for h in range(num_heads):
            for m in range(num_modes):
                try:
                    result = compute_vector_similarity(
                        u_canonical, v_canonical, u_harmonized, v_harmonized, l, h, m)
                    all_similarities[l, h, m] = result['combined_similarity']
                except Exception as e:
                    print(f"Error processing L{l} H{h} M{m}: {e}")
                    all_similarities[l, h, m] = 0.0
    
    layer_avg_similarities = np.mean(all_similarities, axis=(1, 2))
    head_avg_similarities = np.mean(all_similarities, axis=(0, 2))
    mode_avg_similarities = np.mean(all_similarities, axis=(0, 1))
    
    results = {
        'all_similarities': all_similarities,
        'layer_avg_similarities': layer_avg_similarities,
        'head_avg_similarities': head_avg_similarities,
        'mode_avg_similarities': mode_avg_similarities,
        'most_changed_lhm': np.unravel_index(np.argmin(all_similarities), all_similarities.shape),
        'least_changed_lhm': np.unravel_index(np.argmax(all_similarities), all_similarities.shape)
    }
    
    return results


def compare_activations(
    act_canonical: np.ndarray,
    act_harmonized: np.ndarray,
    layer: int = None,
    head: int = None,
    mode: int = None
) -> dict:
    """
    Compare activation patterns between canonical and harmonized models.
    
    Args:
        act_canonical: Activations for canonical model
        act_harmonized: Activations for harmonized model
        layer: Specific layer to analyze (None for all)
        head: Specific head to analyze (None for all)
        mode: Specific mode to analyze (None for all)
        
    Returns:
        Dictionary with activation comparison metrics
    """
    results = {}
    
    # Handle single layer/head/mode case
    if layer is not None and head is not None and mode is not None:
        act_can = act_canonical[layer, head, mode]
        act_harm = act_harmonized[layer, head, mode]
        
        # Calculate basic stats
        mean_diff = np.mean(act_harm) - np.mean(act_can)
        std_diff = np.std(act_harm) - np.std(act_can)
        max_diff = np.max(act_harm) - np.max(act_can)
        
        # Calculate Earth Mover's Distance (1D Wasserstein)
        # First sort the activations
        act_can_sorted = np.sort(act_can)
        act_harm_sorted = np.sort(act_harm)
        
        # Calculate the L1 distance between CDFs
        # (simplified version of EMD for 1D distributions)
        emd = np.mean(np.abs(act_can_sorted - act_harm_sorted))
        
        # Calculate selectivity change
        # (how much the model focuses on top vs average activations)
        selectivity_can = np.mean(np.sort(act_can)[-10:]) / np.mean(act_can)
        selectivity_harm = np.mean(np.sort(act_harm)[-10:]) / np.mean(act_harm)
        selectivity_change = selectivity_harm - selectivity_can
        
        results = {
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            'max_diff': max_diff,
            'emd': emd,
            'selectivity_change': selectivity_change,
            'activation_correlation': np.corrcoef(act_can, act_harm)[0, 1]
        }
        return results
    
    # For all layers/heads/modes
    # Implement aggregation similar to previous functions
    # ...
    return results


def create_mode_transformation_taxonomy(
    s_canonical: np.ndarray,
    s_harmonized: np.ndarray,
    u_canonical: np.ndarray,
    v_canonical: np.ndarray,
    u_harmonized: np.ndarray,
    v_harmonized: np.ndarray,
    act_canonical: np.ndarray = None,
    act_harmonized: np.ndarray = None
) -> dict:
    """
    Categorize modes based on how they transformed after harmonization.
    
    Args:
        s_canonical: Singular values for canonical model
        s_harmonized: Singular values for harmonized model
        u_canonical: U vectors for canonical model
        v_canonical: V vectors for canonical model
        u_harmonized: U vectors for harmonized model
        v_harmonized: V vectors for harmonized model
        act_canonical: Activations for canonical model (optional)
        act_harmonized: Activations for harmonized model (optional)
        
    Returns:
        Dictionary with mode transformation taxonomy
    """
    num_layers, num_heads, num_modes = s_canonical.shape
    
    # Categories of transformations
    amplified = []  # Modes that became more important
    suppressed = []  # Modes that became less important
    redirected = []  # Modes that changed direction significantly
    sharpened = []  # Modes that became more selective
    preserved = []  # Modes that remained largely unchanged
    
    # Thresholds for categorization
    importance_thresh = 0.2  # Relative change in singular value
    direction_thresh = 0.8  # Cosine similarity threshold
    selectivity_thresh = 0.2  # Change in selectivity
    
    for l in range(num_layers):
        for h in range(num_heads):
            for m in range(num_modes):
                # Get relative importance change
                s_can = s_canonical[l, h, m]
                s_harm = s_harmonized[l, h, m]
                
                imp_can = s_can / np.sum(s_canonical[l, h])
                imp_harm = s_harm / np.sum(s_harmonized[l, h])
                
                relative_change = (imp_harm - imp_can) / (imp_can + 1e-10)
                
                # Get direction change
                sim_result = compute_vector_similarity(
                    u_canonical, v_canonical, u_harmonized, v_harmonized, l, h, m)
                direction_change = sim_result['alignment_score']
                
                # Categorize based on transformations
                if direction_change > 1 - direction_thresh:
                    redirected.append((l, h, m))
                elif relative_change > importance_thresh:
                    amplified.append((l, h, m))
                elif relative_change < -importance_thresh:
                    suppressed.append((l, h, m))
                else:
                    # Check if it's preserved or sharpened
                    if act_canonical is not None and act_harmonized is not None:
                        act_result = compare_activations(
                            act_canonical, act_harmonized, l, h, m)
                        if act_result['selectivity_change'] > selectivity_thresh:
                            sharpened.append((l, h, m))
                        else:
                            preserved.append((l, h, m))
                    else:
                        preserved.append((l, h, m))
    
    # Calculate percentages
    total_modes = num_layers * num_heads * num_modes
    results = {
        'amplified': amplified,
        'suppressed': suppressed,
        'redirected': redirected,
        'sharpened': sharpened,
        'preserved': preserved,
        'pct_amplified': len(amplified) / total_modes * 100,
        'pct_suppressed': len(suppressed) / total_modes * 100,
        'pct_redirected': len(redirected) / total_modes * 100,
        'pct_sharpened': len(sharpened) / total_modes * 100,
        'pct_preserved': len(preserved) / total_modes * 100
    }
    
    # Layer-wise breakdown
    layer_breakdown = []
    for l in range(num_layers):
        l_amplified = sum(1 for x in amplified if x[0] == l)
        l_suppressed = sum(1 for x in suppressed if x[0] == l)
        l_redirected = sum(1 for x in redirected if x[0] == l)
        l_sharpened = sum(1 for x in sharpened if x[0] == l)
        l_preserved = sum(1 for x in preserved if x[0] == l)
        
        total_l = num_heads * num_modes
        layer_breakdown.append({
            'layer': l,
            'pct_amplified': l_amplified / total_l * 100,
            'pct_suppressed': l_suppressed / total_l * 100,
            'pct_redirected': l_redirected / total_l * 100,
            'pct_sharpened': l_sharpened / total_l * 100,
            'pct_preserved': l_preserved / total_l * 100
        })
    
    results['layer_breakdown'] = layer_breakdown
    
    return results


def run_mode_comparison_analysis(
    canonical_prefix: str,
    harmonized_prefix: str,
    output_dir: str = None,
    data_dir: str = None
):
    """
    Run a comprehensive comparison between canonical and harmonized model modes.
    
    Args:
        canonical_prefix: File prefix for canonical model files
        harmonized_prefix: File prefix for harmonized model files
        output_dir: Directory to save results (default uses config.FIGURE_PATH)
        data_dir: Directory with data files (default uses config.DATA_PATH)
    """
    from config import DATA_PATH, FIGURE_PATH
    import os
    import json
    import matplotlib.pyplot as plt
    from data_utils import load_numpy_array
    
    # Use default paths if not provided
    if output_dir is None:
        output_dir = os.path.join(FIGURE_PATH, "model_comparison")
    if data_dir is None:
        data_dir = os.path.join(DATA_PATH, "UVS")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data for both models
    print(f"Loading canonical model data with prefix: {canonical_prefix}")
    s_canonical = load_numpy_array(os.path.join(data_dir, f"{canonical_prefix}_S_total.npy"))
    u_canonical = load_numpy_array(os.path.join(data_dir, f"{canonical_prefix}_U_total.npy"))
    v_canonical = load_numpy_array(os.path.join(data_dir, f"{canonical_prefix}_Vt_total.npy"))
    
    print(f"Loading harmonized model data with prefix: {harmonized_prefix}")
    s_harmonized = load_numpy_array(os.path.join(data_dir, f"{harmonized_prefix}_S_total.npy"))
    u_harmonized = load_numpy_array(os.path.join(data_dir, f"{harmonized_prefix}_U_total.npy"))
    v_harmonized = load_numpy_array(os.path.join(data_dir, f"{harmonized_prefix}_Vt_total.npy"))
    
    # Try to load activations if available
    try:
        # Check both in UVS directory and optimal_images directory
        activation_paths = [
            os.path.join(data_dir, f"{canonical_prefix}_activations.npy"),  # Original path
            os.path.join(DATA_PATH, "optimal_images", f"{canonical_prefix}_activations.npy")  # New path
        ]
        
        act_canonical = None
        for path in activation_paths:
            if os.path.exists(path):
                print(f"Found canonical activations at: {path}")
                act_canonical = load_numpy_array(path)
                break
        
        if act_canonical is None:
            print(f"Could not find canonical activations in checked paths: {activation_paths}")
            raise FileNotFoundError("Canonical activation file not found")
            
        # Same for harmonized model
        activation_paths = [
            os.path.join(data_dir, f"{harmonized_prefix}_activations.npy"),  # Original path
            os.path.join(DATA_PATH, "optimal_images", f"{harmonized_prefix}_activations.npy")  # New path
        ]
        
        act_harmonized = None
        for path in activation_paths:
            if os.path.exists(path):
                print(f"Found harmonized activations at: {path}")
                act_harmonized = load_numpy_array(path)
                break
        
        if act_harmonized is None:
            print(f"Could not find harmonized activations in checked paths: {activation_paths}")
            raise FileNotFoundError("Harmonized activation file not found")
            
        has_activations = True
        print("Loaded activation data for both models")
    except Exception as e:
        has_activations = False
        act_canonical = None
        act_harmonized = None
        print(f"Activation data not found: {e}")
        print("Skipping activation-based analyses")
    
    # 1. Compare singular values
    print("Comparing singular values...")
    sv_comparison = compare_singular_values(s_canonical, s_harmonized)
    
    # Plot KL divergence by layer to show where distributions changed most
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(sv_comparison['avg_kl_by_layer'])), sv_comparison['avg_kl_by_layer'])
    plt.xlabel('Layer')
    plt.ylabel('Average KL Divergence')
    plt.title('Change in Singular Value Distribution by Layer')
    plt.savefig(os.path.join(output_dir, 'kl_divergence_by_layer.png'))
    
    # Plot concentration differences
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(sv_comparison['avg_concentration_diff_by_layer'])),
            sv_comparison['avg_concentration_diff_by_layer'])
    plt.xlabel('Layer')
    plt.ylabel('Change in Concentration')
    plt.title('Change in Top-5 Mode Concentration by Layer')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.savefig(os.path.join(output_dir, 'concentration_diff_by_layer.png'))
    
    # 2. Compare vector similarities
    print("Computing vector similarities...")
    vec_comparison = compute_vector_similarity(
        u_canonical, v_canonical, u_harmonized, v_harmonized)
    
    # Plot average similarities by layer
    plt.figure(figsize=(12, 6))
    plt.plot(vec_comparison['layer_avg_similarities'])
    plt.xlabel('Layer')
    plt.ylabel('Average Vector Similarity')
    plt.title('How Much Modes Changed Direction By Layer')
    plt.savefig(os.path.join(output_dir, 'vector_similarity_by_layer.png'))
    
    # 3. Create taxonomy of transformations
    print("Creating mode transformation taxonomy...")
    if has_activations:
        taxonomy = create_mode_transformation_taxonomy(
            s_canonical, s_harmonized,
            u_canonical, v_canonical,
            u_harmonized, v_harmonized,
            act_canonical, act_harmonized)
    else:
        taxonomy = create_mode_transformation_taxonomy(
            s_canonical, s_harmonized,
            u_canonical, v_canonical,
            u_harmonized, v_harmonized)
    
    # Plot taxonomy breakdown
    categories = ['Amplified', 'Suppressed', 'Redirected', 'Sharpened', 'Preserved']
    percentages = [
        taxonomy['pct_amplified'],
        taxonomy['pct_suppressed'],
        taxonomy['pct_redirected'],
        taxonomy['pct_sharpened'],
        taxonomy['pct_preserved']
    ]
    
    plt.figure(figsize=(10, 8))
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']
    explode = (0.05, 0.05, 0.05, 0.05, 0.05)
    
    wedges, texts, autotexts = plt.pie(
        percentages, 
        labels=None,
        autopct='%1.1f%%',
        explode=explode,
        colors=colors,
        shadow=False,
        startangle=90,
        textprops={'fontsize': 14, 'weight': 'bold'}
    )
    
    for autotext in autotexts:
        autotext.set_color('black')
    
    plt.legend(
        wedges,
        categories,
        title="Mode Transformations",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=12
    )
    
    plt.title('Taxonomy of Mode Transformations', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'transformation_taxonomy_pie.png'), bbox_inches='tight')
    
    # Plot layer-wise transformation breakdown
    layer_data = taxonomy['layer_breakdown']
    layers = range(len(layer_data))
    
    plt.figure(figsize=(14, 8))
    width = 0.15
    
    plt.bar([x - width*2 for x in layers], [d['pct_amplified'] for d in layer_data], 
            width=width, label='Amplified')
    plt.bar([x - width for x in layers], [d['pct_suppressed'] for d in layer_data], 
            width=width, label='Suppressed')
    plt.bar([x for x in layers], [d['pct_redirected'] for d in layer_data], 
            width=width, label='Redirected')
    plt.bar([x + width for x in layers], [d['pct_sharpened'] for d in layer_data], 
            width=width, label='Sharpened')
    plt.bar([x + width*2 for x in layers], [d['pct_preserved'] for d in layer_data], 
            width=width, label='Preserved')
    
    plt.xlabel('Layer')
    plt.ylabel('Percentage')
    plt.title('Mode Transformations by Layer')
    plt.xticks(layers)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'transformation_by_layer.png'))
    
    # Save numerical results
    results = {
        'singular_value_comparison': {
            'kl_divergence_by_layer': sv_comparison['avg_kl_by_layer'].tolist(),
            'concentration_diff_by_layer': sv_comparison['avg_concentration_diff_by_layer'].tolist()
        },
        'vector_similarity': {
            'layer_avg_similarities': vec_comparison['layer_avg_similarities'].tolist(),
            'most_changed_lhm': [int(x) for x in vec_comparison['most_changed_lhm']],
            'least_changed_lhm': [int(x) for x in vec_comparison['least_changed_lhm']]
        },
        'transformation_taxonomy': {
            'overall': {
                'pct_amplified': float(taxonomy['pct_amplified']),
                'pct_suppressed': float(taxonomy['pct_suppressed']),
                'pct_redirected': float(taxonomy['pct_redirected']),
                'pct_sharpened': float(taxonomy['pct_sharpened']),
                'pct_preserved': float(taxonomy['pct_preserved'])
            },
            'layer_breakdown': [
                {
                    'layer': int(item['layer']),
                    'pct_amplified': float(item['pct_amplified']),
                    'pct_suppressed': float(item['pct_suppressed']),
                    'pct_redirected': float(item['pct_redirected']),
                    'pct_sharpened': float(item['pct_sharpened']),
                    'pct_preserved': float(item['pct_preserved'])
                }
                for item in taxonomy['layer_breakdown']
            ]
        }
    }
    
    # custom JSON encoder class for NumPy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)
    
    with open(os.path.join(output_dir, 'mode_comparison_results.json'), 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    print(f"Analysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare canonical and harmonized model modes")
    parser.add_argument("--canonical_prefix", type=str, default="vit_b16_canonical",
                        help="File prefix for canonical model")
    parser.add_argument("--harmonized_prefix", type=str, default="vit_b16_harmonized",
                        help="File prefix for harmonized model")
    parser.add_argument("--output_dir", type=str, help="Directory to save results")
    parser.add_argument("--data_dir", type=str, help="Directory with data files")
    
    args = parser.parse_args()
    
    run_mode_comparison_analysis(
        args.canonical_prefix,
        args.harmonized_prefix,
        args.output_dir,
        args.data_dir
    )