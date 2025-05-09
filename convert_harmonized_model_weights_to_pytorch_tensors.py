import h5py
import torch
import numpy as np
from transformers import ViTForImageClassification, ViTConfig

# Create model architecture with 1000 output classes
config = ViTConfig(
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    num_labels=1000
)
model = ViTForImageClassification(config)

# Load the H5 file with the specific filename
h5_file_path = 'vit-b16_harmonized.h5'
with h5py.File(h5_file_path, 'r') as f:
    # Convert H5 weights to PyTorch state dict
    h5_state_dict = {}
    
    # Iterate through the H5 file structure
    def recursively_load_weights(h5_group, state_dict_path=''):
        for key, item in h5_group.items():
            path = state_dict_path + '.' + key if state_dict_path else key
            
            if isinstance(item, h5py.Group):
                recursively_load_weights(item, path)
            else:
                # Handle different data types
                data = item[()]
                if isinstance(data, (np.number, int, float)):
                    h5_state_dict[path] = torch.tensor(data)
                else:
                    h5_state_dict[path] = torch.from_numpy(data)
    
    # Start recursive loading from the root
    recursively_load_weights(f)

# Create a more comprehensive direct mapping for layers 10 and 11
direct_mapping = {}

# Print new missing keys
print("Missing keys to map:")
for k in ['vit.encoder.layer.11.attention.attention.value.weight', 
          'vit.encoder.layer.11.attention.attention.query.bias',
          'vit.encoder.layer.11.attention.attention.key.bias',
          'vit.encoder.layer.10.intermediate.dense.weight',
          'vit.encoder.layer.11.attention.attention.query.weight']:
    print(f"  {k}")

# List all keys for layers 10 and 11
layer10_11_keys = [k for k in h5_state_dict.keys() 
                  if ('encoderblock_10' in k or 'encoderblock_11' in k) 
                  and 'model_weights' in k 
                  and not 'optimizer' in k]

# Complete mapping for layer 10
for h5_key in layer10_11_keys:
    if ':0' in h5_key:
        base_key = h5_key.split(':0')[0]
    else:
        base_key = h5_key
        
    # Layer 10 mappings
    if 'encoderblock_10' in base_key:
        if 'MultiHeadDotProductAttention_1.query.kernel' in base_key:
            direct_mapping[base_key] = 'vit.encoder.layer.10.attention.attention.query.weight'
        elif 'MultiHeadDotProductAttention_1.query.bias' in base_key:
            direct_mapping[base_key] = 'vit.encoder.layer.10.attention.attention.query.bias'
        elif 'MultiHeadDotProductAttention_1.key.kernel' in base_key:
            direct_mapping[base_key] = 'vit.encoder.layer.10.attention.attention.key.weight'
        elif 'MultiHeadDotProductAttention_1.key.bias' in base_key:
            direct_mapping[base_key] = 'vit.encoder.layer.10.attention.attention.key.bias'
        elif 'MultiHeadDotProductAttention_1.value.kernel' in base_key:
            direct_mapping[base_key] = 'vit.encoder.layer.10.attention.attention.value.weight'
        elif 'MultiHeadDotProductAttention_1.value.bias' in base_key:
            direct_mapping[base_key] = 'vit.encoder.layer.10.attention.attention.value.bias'
        elif 'MultiHeadDotProductAttention_1.out.kernel' in base_key:
            direct_mapping[base_key] = 'vit.encoder.layer.10.attention.output.dense.weight'
        elif 'MultiHeadDotProductAttention_1.out.bias' in base_key:
            direct_mapping[base_key] = 'vit.encoder.layer.10.attention.output.dense.bias'
        elif 'Dense_0.kernel' in base_key:
            direct_mapping[base_key] = 'vit.encoder.layer.10.intermediate.dense.weight'
        elif 'Dense_0.bias' in base_key:
            direct_mapping[base_key] = 'vit.encoder.layer.10.intermediate.dense.bias'
        elif 'Dense_1.kernel' in base_key:
            direct_mapping[base_key] = 'vit.encoder.layer.10.output.dense.weight'
        elif 'Dense_1.bias' in base_key:
            direct_mapping[base_key] = 'vit.encoder.layer.10.output.dense.bias'
        elif 'LayerNorm_0.gamma' in base_key:
            direct_mapping[base_key] = 'vit.encoder.layer.10.layernorm_before.weight'
        elif 'LayerNorm_0.beta' in base_key:
            direct_mapping[base_key] = 'vit.encoder.layer.10.layernorm_before.bias'
        elif 'LayerNorm_2.gamma' in base_key:
            direct_mapping[base_key] = 'vit.encoder.layer.10.layernorm_after.weight'
        elif 'LayerNorm_2.beta' in base_key:
            direct_mapping[base_key] = 'vit.encoder.layer.10.layernorm_after.bias'
            
    # Layer 11 mappings
    if 'encoderblock_11' in base_key:
        if 'MultiHeadDotProductAttention_1.query.kernel' in base_key:
            direct_mapping[base_key] = 'vit.encoder.layer.11.attention.attention.query.weight'
        elif 'MultiHeadDotProductAttention_1.query.bias' in base_key:
            direct_mapping[base_key] = 'vit.encoder.layer.11.attention.attention.query.bias'
        elif 'MultiHeadDotProductAttention_1.key.kernel' in base_key:
            direct_mapping[base_key] = 'vit.encoder.layer.11.attention.attention.key.weight'
        elif 'MultiHeadDotProductAttention_1.key.bias' in base_key:
            direct_mapping[base_key] = 'vit.encoder.layer.11.attention.attention.key.bias'
        elif 'MultiHeadDotProductAttention_1.value.kernel' in base_key:
            direct_mapping[base_key] = 'vit.encoder.layer.11.attention.attention.value.weight'
        elif 'MultiHeadDotProductAttention_1.value.bias' in base_key:
            direct_mapping[base_key] = 'vit.encoder.layer.11.attention.attention.value.bias'
        elif 'MultiHeadDotProductAttention_1.out.kernel' in base_key:
            direct_mapping[base_key] = 'vit.encoder.layer.11.attention.output.dense.weight'
        elif 'MultiHeadDotProductAttention_1.out.bias' in base_key:
            direct_mapping[base_key] = 'vit.encoder.layer.11.attention.output.dense.bias'
        elif 'Dense_0.kernel' in base_key:
            direct_mapping[base_key] = 'vit.encoder.layer.11.intermediate.dense.weight'
        elif 'Dense_0.bias' in base_key:
            direct_mapping[base_key] = 'vit.encoder.layer.11.intermediate.dense.bias'
        elif 'Dense_1.kernel' in base_key:
            direct_mapping[base_key] = 'vit.encoder.layer.11.output.dense.weight'
        elif 'Dense_1.bias' in base_key:
            direct_mapping[base_key] = 'vit.encoder.layer.11.output.dense.bias'
        elif 'LayerNorm_0.gamma' in base_key:
            direct_mapping[base_key] = 'vit.encoder.layer.11.layernorm_before.weight'
        elif 'LayerNorm_0.beta' in base_key:
            direct_mapping[base_key] = 'vit.encoder.layer.11.layernorm_before.bias'
        elif 'LayerNorm_2.gamma' in base_key:
            direct_mapping[base_key] = 'vit.encoder.layer.11.layernorm_after.weight'
        elif 'LayerNorm_2.beta' in base_key:
            direct_mapping[base_key] = 'vit.encoder.layer.11.layernorm_after.bias'

# Now create our standard mapping function
def map_key(h5_key):
    # Skip optimizer weights
    if 'optimizer_weights' in h5_key:
        return None
        
    # Get base key without :0
    if ':0' in h5_key:
        base_key = h5_key.split(':0')[0]
    else:
        base_key = h5_key

    # Check if we have a direct mapping
    if base_key in direct_mapping:
        return direct_mapping[base_key]
    
    # Map embeddings
    if 'embedding.kernel' in base_key:
        return 'vit.embeddings.patch_embeddings.projection.weight'
    elif 'embedding.bias' in base_key:
        return 'vit.embeddings.patch_embeddings.projection.bias'
    elif 'class_token.cls' in base_key:
        return 'vit.embeddings.cls_token'
    elif 'posembed_input.pos_embedding' in base_key:
        return 'vit.embeddings.position_embeddings'
    
    # Map encoder norm
    elif 'encoder_norm.gamma' in base_key:
        return 'vit.layernorm.weight'
    elif 'encoder_norm.beta' in base_key:
        return 'vit.layernorm.bias'
    
    # Map head
    elif 'head.kernel' in base_key:
        return 'classifier.weight'
    elif 'head.bias' in base_key:
        return 'classifier.bias'
    
    # Map encoder blocks for layers 0-9
    elif 'encoderblock_' in base_key:
        # Find layer number
        layer_num = None
        for i in range(10):  # Only process layers 0-9 here
            if f'encoderblock_{i}.' in base_key:
                layer_num = i
                break
                
        if layer_num is None:
            return None
            
        # Map attention components
        if 'MultiHeadDotProductAttention_1.query.kernel' in base_key:
            return f'vit.encoder.layer.{layer_num}.attention.attention.query.weight'
        elif 'MultiHeadDotProductAttention_1.query.bias' in base_key:
            return f'vit.encoder.layer.{layer_num}.attention.attention.query.bias'
        elif 'MultiHeadDotProductAttention_1.key.kernel' in base_key:
            return f'vit.encoder.layer.{layer_num}.attention.attention.key.weight'
        elif 'MultiHeadDotProductAttention_1.key.bias' in base_key:
            return f'vit.encoder.layer.{layer_num}.attention.attention.key.bias'
        elif 'MultiHeadDotProductAttention_1.value.kernel' in base_key:
            return f'vit.encoder.layer.{layer_num}.attention.attention.value.weight'
        elif 'MultiHeadDotProductAttention_1.value.bias' in base_key:
            return f'vit.encoder.layer.{layer_num}.attention.attention.value.bias'
        elif 'MultiHeadDotProductAttention_1.out.kernel' in base_key:
            return f'vit.encoder.layer.{layer_num}.attention.output.dense.weight'
        elif 'MultiHeadDotProductAttention_1.out.bias' in base_key:
            return f'vit.encoder.layer.{layer_num}.attention.output.dense.bias'
        
        # Map MLP components
        elif 'Dense_0.kernel' in base_key:
            return f'vit.encoder.layer.{layer_num}.intermediate.dense.weight'
        elif 'Dense_0.bias' in base_key:
            return f'vit.encoder.layer.{layer_num}.intermediate.dense.bias'
        elif 'Dense_1.kernel' in base_key:
            return f'vit.encoder.layer.{layer_num}.output.dense.weight'
        elif 'Dense_1.bias' in base_key:
            return f'vit.encoder.layer.{layer_num}.output.dense.bias'
        
        # Map LayerNorms
        elif 'LayerNorm_0.gamma' in base_key:
            return f'vit.encoder.layer.{layer_num}.layernorm_before.weight'
        elif 'LayerNorm_0.beta' in base_key:
            return f'vit.encoder.layer.{layer_num}.layernorm_before.bias'
        elif 'LayerNorm_2.gamma' in base_key:
            return f'vit.encoder.layer.{layer_num}.layernorm_after.weight'
        elif 'LayerNorm_2.beta' in base_key:
            return f'vit.encoder.layer.{layer_num}.layernorm_after.bias'
    
    return None

# Create a new state dict with correctly mapped keys
remapped_state_dict = {}
for h5_key, value in h5_state_dict.items():
    # Get the mapped key
    model_key = map_key(h5_key)
    
    if model_key is not None:
        # Transpose weights for linear layers if needed
        if ('kernel' in h5_key and 'weight' in model_key and 
            ('dense' in model_key or 'projection' in model_key or 'classifier' in model_key)):
            # For classifier.weight, the shape should be [num_classes, hidden_size]
            if model_key == 'classifier.weight':
                # Check if we need to transpose or not based on shapes
                if value.shape[0] == 1000:  # Already in the right orientation
                    pass
                elif value.shape[1] == 1000:  # Need transposing
                    value = value.T
            else:
                value = value.T
            
        remapped_state_dict[model_key] = value

# Check mapping status
print(f"\nSuccessfully mapped {len(remapped_state_dict)} keys")
model_keys = set(model.state_dict().keys())
remapped_keys = set(remapped_state_dict.keys())
print("Missing in remapped dict:", len(model_keys - remapped_keys))
if len(model_keys - remapped_keys) > 0:
    print("Still missing keys:", list(model_keys - remapped_keys)[:5])
print("Extra in remapped dict:", len(remapped_keys - model_keys))

# Load weights into model with strict=False to ignore missing keys
model.load_state_dict(remapped_state_dict, strict=False)
print("Model loaded successfully!")

# Save the loaded model
torch.save(model.state_dict(), "vit_b16_harmonized_pytorch.pth")
print("Model saved as vit_b16_harmonized_pytorch.pth")

# Print the specific keys that have been mapped for layers 10 and 11
mapped_layer10_11_keys = [k for k in remapped_state_dict.keys() if 'layer.10' in k or 'layer.11' in k]
print(f"\nSuccessfully mapped {len(mapped_layer10_11_keys)} keys for layers 10 and 11:")
for k in sorted(mapped_layer10_11_keys):
    print(f"  {k}")