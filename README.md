# DissectingViT

This repository contains refactored code for the DissectingViT analysis pipeline. The code is organized into modular components that can be run as standalone scripts or as part of a complete pipeline.

## Overview

The DissectingViT pipeline consists of the following steps:

1. **SVD Computation**: Compute the Singular Value Decomposition (SVD) of attention matrices and save U, V, and S matrices.
2. **Cosine Similarity Analysis**: Analyze and visualize the cosine similarity between U and V matrices.
3. **Activation Computation**: Compute activations for each layer, head, and mode using a dataset of images.
4. **Visualization**: Generate heatmap visualizations for optimal images based on activations.

## Setup

The code is designed to work with Python 3.7+ and requires the following packages:

```bash
pip install torch numpy matplotlib transformers pillow scipy
```

## Directory Structure

- `config.py`: Configuration settings for paths, models, and devices.
- `data_utils.py`: Utilities for data loading and manipulation.
- `logging_utils.py`: Utilities for logging configuration.
- `model_loader.py`: Functions for loading and initializing models.
- `model_analysis.py`: Classes and functions for analyzing models.
- `visualization.py`: Functions for visualization.
- `compute_svd.py`: Script to compute SVD for models.
- `plot_cosine_similarity.py`: Script to generate cosine similarity plots.
- `compute_optimal_images.py`: Script to compute activations and visualize optimal images.
- `main.py`: Main script to run the complete pipeline.

## Usage

### Running the Complete Pipeline

To run the complete pipeline with default settings:

```bash
python main.py --models "facebook/dino-vits16"
```

To customize the pipeline:

```bash
python main.py --models "facebook/dino-vits16" "google/vit-base-patch16-224" --no_svd --layer 3 --head 2 --modes 0 1 2
```

### Running Individual Steps

#### 1. Compute SVD

```bash
python compute_svd.py --models "facebook/dino-vits16"
```

#### 2. Generate Cosine Similarity Plots

```bash
python plot_cosine_similarity.py --models "facebook/dino-vits16"
```

#### 3. Compute Activations

```bash
python compute_optimal_images.py --model "facebook/dino-vits16" --compute_only
```

#### 4. Visualize Optimal Images

```bash
python compute_optimal_images.py --model "facebook/dino-vits16" --layer 3 --head 2 --sign "p" --modes 0 1 2 3 4
```

## Command Line Arguments

### Main Script

- `--models`: List of model names to process (default: from config.py)
- `--data_path`: Path to the data directory (default: from config.py)
- `--figure_path`: Path to save the output figures (default: from config.py)
- `--dataset_path`: Path to the dataset (default: from config.py)
- `--no_svd`: Skip SVD computation
- `--no_cosine`: Skip cosine similarity plots
- `--no_activations`: Skip activation computation
- `--no_visualizations`: Skip visualizations
- `--layer`: Layer index for visualizations (default: 0)
- `--head`: Head index for visualizations (default: 0)
- `--sign`: Sign of the activations ('p' or 'n') (default: 'p')
- `--modes`: List of mode indices to visualize (default: [0, 1, 2, 3, 4])
- `--log_dir`: Directory for log files (default: BASE_DIR/logs)

## Logging

All scripts include comprehensive logging to both console and log files. Log files are saved in the specified `log_dir` with timestamps in the filename for easy tracking.

## Examples

### Example 1: Complete Pipeline for a Single Model

```bash
python main.py --models "facebook/dino-vits16" --dataset_path "/path/to/imagenette/val"
```

### Example 2: Generate Visualizations Only

```bash
python main.py --models "facebook/dino-vits16" --no_svd --no_cosine --no_activations --layer 5 --head 7 --sign "n" --modes 0 1 2
```

### Example 3: Run SVD on Multiple Models

```bash
python compute_svd.py --models "facebook/dino-vits16" "google/vit-base-patch16-224" "openai/clip-vit-base-patch16"
```

## Configuration

Edit `config.py` to change default paths and settings. The most important settings are:

- `BASE_DIR`: Base directory for the project
- `DATA_PATH`: Path to save data files
- `FIGURE_PATH`: Path to save figure files
- `IMAGENET_VAL_PATH`: Path to validation images
- `DEFAULT_MODELS`: List of default models to process
- `USE_GPU`: Whether to use GPU if available
