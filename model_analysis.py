# models/model_analysis.py
"""
Tools for analyzing vision transformer models.
"""
from typing import Any, List, Tuple

import numpy as np
import torch


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

    def hook_fn(
        self, module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor
    ) -> None:
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

    def forward(
        self,
        input: torch.Tensor,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> Any:
        """Forward pass through the model."""
        with torch.no_grad():
            output = self.model(
                input,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )
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

    def hook_fn(
        self, module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor
    ) -> None:
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

    def forward(
        self,
        input: torch.Tensor,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> Any:
        """Forward pass through the model."""
        with torch.no_grad():
            output = self.model(
                input,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )
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

    def hook_fn(
        self, module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor
    ) -> None:
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

    def forward(
        self,
        input: torch.Tensor,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> Any:
        """Forward pass through the model."""
        with torch.no_grad():
            output = self.model(
                input,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )
        return output


class ModelAnalyzer:
    """Tools for analyzing transformer models."""

    def __init__(self, model: torch.nn.Module):
        """
        Initialize analyzer with a model.

        Args:
            model: Model to analyze
        """
        self.layer_list = [
            (name, param.to(torch.device("cpu")))
            for name, param in model.named_parameters()
        ]

    def block_diagonalize(
        self, A: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
                block_diag[2 * i + find0, 2 * i + 1 + find0] = imageigenvalues[
                    2 * i + find0
                ]
                block_diag[2 * i + 1 + find0, 2 * i + find0] = imageigenvalues[
                    2 * i + 1 + find0
                ]
                V.append(realeigenvectors[:, 2 * i + find0] * np.sqrt(2))
                V.append(imageigenvectors[:, 2 * i + find0] * np.sqrt(2))
            if imageigenvalues[d - 1] == 0:  # check boundary condition
                block_diag[d - 1, d - 1] = 0
                V.append(realeigenvectors[:, d - 1])

        V = np.array(V)
        V = V.T

        return block_diag, V, imageigenvalues

    def svd_angles(
        self, M: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
