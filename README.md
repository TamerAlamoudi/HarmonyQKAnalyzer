# HarmonyQKAnalyzer

This repository contains code for the HarmonyQKAnalyzer framework for analyzing Vision Transformer (ViT) models by decomposing their attention mechanisms, with a focus on how harmonization affects query-key interactions.

## Setup and Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure all scripts have execute permissions:
```bash
chmod +x *.py
```

## Downloading and Converting Harmonized Model Weights

To work with harmonized model weights:

1. Download the harmonized ViT model weights:
```bash
wget https://storage.googleapis.com/serrelab/prj_harmonization/models/vit-b16_harmonized.h5
```

2. Convert the weights to PyTorch format:
```bash
python convert_harmonized_weights_to_pytorch_tensors.py
```

This will generate a `vit_b16_harmonized_pytorch.pth` file that can be used with the framework.

## Configuration

Configuration parameters are centralized in `config.py`. The main settings include:

### Paths
- `BASE_DIR`: Base directory for the project (default: `/Users/tamer/research`)
- `DATA_PATH`: Directory for saving data and results
- `FIGURE_PATH`: Directory for saving output figures
- `LOGS_PATH`: Directory for saving log files
- `IMAGENET_VAL_PATH`: Path to the ImageNet validation dataset

### Device Configuration
- `DEVICE`: Computing device ('cpu', 'cuda', or 'mps')
- `USE_GPU`: Boolean flag to enable GPU usage when available

### Models
- `DEFAULT_MODELS`: List of model names to use when none are specified

You can override most of these settings using command-line arguments when running the scripts.

## File Naming and Consistency

When working with custom models, the framework generates file names based on the model name and weights file. To ensure consistency between different steps of the pipeline, you can use the `--file_prefix` parameter to explicitly specify the file prefix to use:

```bash
# Step 1: Compute SVD with a specific file prefix
python compute_svd.py --models google/vit-base-patch16-224 --weights_paths ./models/my_weights.pth --file_prefixes my_custom_model

# Step 2: Compute activations using the same file prefix
python compute_optimal_images.py --model google/vit-base-patch16-224 --weights_path ./models/my_weights.pth --file_prefix my_custom_model
```

This ensures that files are consistently named across all steps of the pipeline, making it easier to track your analysis.

## Project Structure

- **Core Modules**:
  - `config.py`: Central configuration settings
  - `data_utils.py`: Data loading and manipulation utilities
  - `logging_utils.py`: Configurable logging setup
  - `model_loader.py`: Functions for loading various ViT model types
  - `model_analysis.py`: Model analysis classes with hook mechanisms
  - `visualization.py`: Visualization functions

- **Execution Scripts**:
  - `compute_svd.py`: Computes SVD for Q and K matrices
  - `plot_cosine_similarity.py`: Generates cosine similarity plots
  - `compute_optimal_images.py`: Computes activations and visualizes optimal images
  - `main.py`: Runs the complete pipeline in sequence
  - `custom_model_example.py`: Example for using custom model weights

- **Utility Scripts**:
  - `utils/convert_weights.py`: Converts weights from different formats to be compatible with HarmonyQKAnalyzer

## Pipeline Steps - Detailed Guide

The analysis pipeline consists of several sequential steps, each of which can be run independently or as part of the full pipeline using the `main.py` script.

### 1. Computing SVD

The first step in the pipeline is computing the Singular Value Decomposition (SVD) of the Query and Key matrices for each attention head.

#### Pre-trained Models from Hugging Face

```bash
# Compute SVD for a single pre-trained model
python compute_svd.py --models google/vit-base-patch16-224

# Compute SVD for multiple models
python compute_svd.py --models google/vit-base-patch16-224 facebook/dino-vits16
```

#### Custom Model Weights

```bash
# Compute SVD for a custom model
python compute_svd.py --models my_custom_model --weights_paths ./models/my_vit.pth --config_files ./models/my_vit_config.json

# Use a specific file prefix for consistency
python compute_svd.py --models google/vit-base-patch16-224 --weights_paths ./models/my_vit.pth --file_prefixes google_vit_harmonized
```

#### Using main.py with Skip Flags

```bash
# Run only the SVD computation step using main.py
python main.py --model_name google/vit-base-patch16-224 --skip_cosine --skip_activations --skip_visualizations
```

### 2. Plotting Cosine Similarity

This step generates plots showing the cosine similarity between singular vectors.

#### Pre-trained Models

```bash
# Generate cosine similarity plots for a pre-trained model
python plot_cosine_similarity.py --model_name google/vit-base-patch16-224

# Specify a custom figure output path
python plot_cosine_similarity.py --model_name facebook/dino-vits16 --figure_path ./custom_figures
```

#### Custom Models

```bash
# Generate cosine similarity plots for a custom model
python plot_cosine_similarity.py --model_name my_custom_model --weights_path ./models/my_vit.pth

# Use a specific file prefix for consistency
python plot_cosine_similarity.py --model_name google/vit-base-patch16-224 --weights_path ./models/my_vit.pth --file_prefix google_vit_harmonized
```

#### Using main.py with Skip Flags

```bash
# Run only the SVD and cosine similarity steps
python main.py --model_name google/vit-base-patch16-224 --skip_activations --skip_visualizations
```

### 3. Computing Activations and Optimal Images

This step computes the activations of each singular vector on the dataset images and identifies the optimal images for each mode.

#### Pre-trained Models

```bash
# Compute activations for a pre-trained model
python compute_optimal_images.py --model google/vit-base-patch16-224

# Specify a particular dataset path
python compute_optimal_images.py --model google/vit-base-patch16-224 --dataset_path /path/to/dataset
```

#### Custom Models

```bash
# Compute activations for a custom model
python compute_optimal_images.py --model my_custom_model --weights_path ./models/my_vit.pth

# Use a specific file prefix to match SVD files
python compute_optimal_images.py --model google/vit-base-patch16-224 --weights_path ./models/my_vit.pth --file_prefix google_vit_harmonized
```

#### Compute Activations for Specific Layers/Heads

```bash
# Compute activations only for a specific layer and head
python compute_optimal_images.py --model google/vit-base-patch16-224 --layer 5 --head 3
```

### 4. Visualization

The visualization step generates heatmaps and other visualizations of the most activated images for each mode.

#### Basic Visualization

```bash
# Visualize optimal images for a pre-trained model
python compute_optimal_images.py --model_name google/vit-base-patch16-224 --skip_compute_activations

# Visualize specific modes
python compute_optimal_images.py --model_name google/vit-base-patch16-224 --skip_compute_activations --layer 3 --head 2 --sign p --modes 0 1 2 3 4
```

#### Custom Model Visualization

```bash
# Visualize optimal images for a custom model
python compute_optimal_images.py --model_name my_custom_model --weights_path ./models/my_vit.pth --skip_compute_activations
```

## Full Pipeline Examples

### Run the Complete Pipeline for Pre-trained Models

The `main.py` script can be used to run the full pipeline:

```bash
# Run the full pipeline for a single pre-trained model
python main.py --model_name google/vit-base-patch16-224

# Run the full pipeline for multiple models
python main.py --models google/vit-base-patch16-224 facebook/dino-vits16
```

### Run the Complete Pipeline for Custom Models

```bash
# Run the full pipeline for a custom model
python main.py --model_name custom/my_model --weights_path ./models/my_model.pth --config_file ./models/my_config.json
```

### Selective Pipeline Execution

You can use skip flags to run only parts of the pipeline:

```bash
# Skip SVD computation (useful if SVD matrices are already computed)
python main.py --model_name google/vit-base-patch16-224 --skip_svd

# Run only visualization for specific layers/heads
python main.py --model_name google/vit-base-patch16-224 --skip_svd --skip_cosine --skip_activations --layer 3 --head 2 --sign p --modes 0 1 2 3 4
```

## Custom Model Analysis

The `custom_model_example.py` script demonstrates how to load and analyze a model with custom weights:

```bash
python custom_model_example.py --weights_path /path/to/weights.pth --model_name my_custom_model
```

### Required Parameters:
- `--weights_path`: Path to custom model weights file (.pth, .pt, or .bin)

### Optional Parameters:
- `--model_type`: Type of model architecture (vit, deit, clip) [default: vit]
- `--config_file`: Path to JSON configuration file for the model
- `--model_name`: Name to use for this model in files and logs [default: custom_model]
- `--data_path`: Path to save the results [default: from config]
- `--figure_path`: Path to save output figures [default: from config]
- `--dataset_path`: Path to the dataset [default: from config]
- `--log_dir`: Directory for log files [default: BASE_DIR/logs]
- `--skip_svd`: Skip SVD computation (use if already computed)

### Visualization Parameters:
- `--layer`: Specific layer to visualize (if not specified, processes all layers)
- `--head`: Specific head to visualize (if not specified, processes all heads)
- `--sign`: Sign to visualize, "p" or "n" (if not specified, processes both)
- `--modes`: List of mode indices to visualize (if not specified, uses top 20 modes)

### Example:

```bash
python custom_model_example.py --weights_path ./models/my_vit.pth --config_file ./models/my_vit_config.json --model_name my_vit --layer 5 --head 3 --sign p
```

## Converting Custom Model Weights

If you have model weights in a different format, you can use the `utils/convert_weights.py` script to convert them:

```bash
# Convert PyTorch weights
python utils/convert_weights.py --input_path ./original_weights.pth --output_path ./converted_weights.pth --model_type vit

# Convert from Hugging Face model
python utils/convert_weights.py --input_path google/vit-base-patch16-224 --output_path ./converted_weights.pth --from_huggingface
```

## Output Structure

After running the pipeline, the following outputs will be generated:

- `{DATA_PATH}/UVS/`: Contains SVD matrices for each model
- `{DATA_PATH}/optimal_images/`: Contains activation data for optimal images
- `{FIGURE_PATH}/cosine/`: Contains cosine similarity plots
- `{FIGURE_PATH}/optimal_images/`: Contains visualizations of optimal images
- `{LOGS_PATH}/`: Contains log files from pipeline execution 

## Model Comparison Analysis

The framework includes comprehensive tools for comparing different ViT models, particularly for analyzing the differences between canonical and harmonized models. These tools help understand how model modifications affect attention mechanisms.

### Running Model Comparison

The `model_analysis.py` script provides functions for analyzing and comparing models:

```bash
# Compare canonical and harmonized models
python model_analysis.py --canonical_prefix vit_b16_canonical --harmonized_prefix vit_b16_harmonized --output_dir model_comparison
```

### Available Comparison Metrics

The analysis pipeline generates several metrics to quantify differences between models:

1. **KL Divergence by Layer**: Measures how the distribution of singular values has changed between models. Higher values indicate more significant redistributions of attention.

2. **Concentration Differences**: Quantifies how focused attention is on the top modes. Positive values indicate increased concentration (more selective attention), while negative values suggest more distributed attention.

3. **Vector Similarity**: Measures the cosine similarity between corresponding singular vectors in different models. Lower values indicate more substantial redirection of attention patterns.

4. **Transformation Taxonomy**: Categorizes mode transformations into:
   - **Redirected**: Modes whose attention directions change significantly
   - **Amplified**: Modes that become more important after transformation
   - **Suppressed**: Modes that become less important
   - **Sharpened**: Modes that become more selective in their activation
   - **Preserved**: Modes that remain largely unchanged

### Visualization Outputs

The comparison analysis generates several visualizations:

```bash
# Generate all comparison visualizations
python compare_models.py --model1 vit_b16_canonical --model2 vit_b16_harmonized --output_dir model_comparison
```

1. **Transformation Taxonomy Pie Chart**: Shows the percentage breakdown of how modes were transformed.
   
2. **Transformation by Layer**: Bar chart showing the distribution of transformation types across layers.
   
3. **KL Divergence by Layer**: Line plot showing the layer-wise distribution changes.
   
4. **Concentration Difference by Layer**: Shows how attention concentration changes across layers.
   
5. **Vector Similarity by Layer**: Visualizes how much attention directions change across layers.
   
6. **Weighted Cosine Similarity Comparison**: Compares the cosine similarity patterns between left and right singular vectors across layers for both models.

### Example: Comparing Canonical and Harmonized Models

```bash
# Step 1: Run SVD computation for both models
python compute_svd.py --models google/vit-base-patch16-224 --file_prefixes vit_b16_canonical
python compute_svd.py --models google/vit-base-patch16-224 --weights_paths ./models/harmonized_vit.pth --file_prefixes vit_b16_harmonized

# Step 2: Run comparison analysis
python model_analysis.py --canonical_prefix vit_b16_canonical --harmonized_prefix vit_b16_harmonized --output_dir model_comparison

# Step 3: Generate comparison plots
python compare_models.py --model1 vit_b16_canonical --model2 vit_b16_harmonized --output_dir model_comparison --file_prefixes canonical harmonized
```

### Interpreting Results

The comparison metrics help interpret how model modifications affect attention:

- **Early Layer Diversification**: Negative concentration changes in early layers suggest more diverse feature extraction.
- **Late Layer Selective Focusing**: Positive concentration changes in later layers indicate more selective attention to important features.
- **Complete Redirection**: Low vector similarity values (near 0.1) indicate almost complete rewiring of attention mechanisms.
- **Layer-Specific Changes**: KL divergence peaks identify layers that underwent the most significant changes in attention distribution.

These metrics provide a mechanistic explanation of how model modifications like harmonization alter the internal processing of visual information. 