# QLoRA: Efficient Fine-tuning with 4-bit Quantization

A PyTorch implementation of QLoRA (Quantized Low-Rank Adaptation) using BitsAndBytes 4-bit quantization for memory-efficient fine-tuning of large language models.

## Overview

This project demonstrates how to efficiently fine-tune large language models using a unified QLoRA/LoRA implementation that combines:
- **4-bit Quantization** via BitsAndBytes to reduce memory footprint (optional)
- **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning
- **Flexible Architecture**: Switch between QLoRA (4-bit) and standard LoRA with a single parameter

The same `Qlora_4_bit` class can operate in multiple modes:
- **QLoRA Mode** (`use_qlora=True`): 4-bit quantized base + LoRA adapters
- **LoRA Mode** (`use_qlora=False`): Standard FP16/BF16 base + LoRA adapters
- **Quantization Only** (`require_lora=False`): Just 4-bit quantization for inference

This enables fine-tuning of billion-parameter models on consumer GPUs by reducing memory requirements by up to 4x while maintaining performance.

## Key Features

- **Unified QLoRA/LoRA Layer**: Single class supports both 4-bit quantized and standard modes
- **Flexible Modes**: 
  - QLoRA: 4-bit quantization + LoRA adapters
  - LoRA: Standard precision + LoRA adapters  
  - Quantization-only: For inference without adapters
- **BitsAndBytes Integration**: Leverages `bnb.nn.Linear4bit` for efficient quantization
- **Selective Fine-tuning**: Target specific layers (MLP, attention, etc.)
- **Memory Efficient**: Fine-tune 1B+ parameter models on limited VRAM
- **Easy Integration**: Drop-in replacement for `nn.Linear` layers

## Technologies Used

- **PyTorch**: Deep learning framework
- **BitsAndBytes**: 4-bit quantization library
- **Transformers**: Hugging Face model hub integration
- **Datasets**: Dataset loading and preprocessing

## Quick Start

### Basic Usage

```python
import torch
from qlora_layer import Qlora_4_bit, quantize_model
from transformers import AutoModelForCausalLM

# Load your model
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_map="auto",
    torch_dtype=torch.float16
)

# Define which layers to apply QLoRA to
target_layers = ["gate_proj", "up_proj", "down_proj"]  # MLP layers

# Apply QLoRA quantization
quantize_model(
    module=model,
    mlp_layers=target_layers,
    alpha=4,              # LoRA scaling factor (alpha/r ~ 2)
    r=2,                  # LoRA rank
    require_lora=True,    # Enable LoRA adapters
    use_qlora=True        # Use 4-bit quantization
)

# Freeze non-LoRA parameters
model.model.embed_tokens.weight.requires_grad = False
```

### Training Example

```python
from transformers import TrainingArguments, Trainer
from datasets import load_dataset

# Load dataset
dataset = load_dataset("Abirate/english_quotes")

# Configure training
training_args = TrainingArguments(
    output_dir="./qlora_outputs",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_steps=60,
    logging_steps=5,
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
)

trainer.train()
```

## How It Works

### Unified QLoRA/LoRA Layer Architecture

The `Qlora_4_bit` class is a flexible implementation that adapts based on configuration:

**Architecture:**
```
Input → [Base Weight Layer] → Output₁
    ↓
    → [LoRA: W_A × W_B (trainable)] × α/r → Output₂
    
Final Output = Output₁ + Output₂
```

**Mode Selection:**
- `use_qlora=True`: Base layer uses `bnb.nn.Linear4bit` (4-bit quantized)
- `use_qlora=False`: Base layer uses standard `nn.Linear` (FP16/BF16)
- `require_lora=True`: Adds LoRA adapters W_A and W_B
- `require_lora=False`: Only uses base layer (useful for inference)

**Key Components:**
1. **Base Layer**: Either 4-bit quantized (`use_qlora=True`) or standard linear layer
2. **LoRA Adapters** (when `require_lora=True`): 
   - `W_A`: Input dimension × rank (initialized with Kaiming uniform)
   - `W_B`: Rank × output dimension (initialized to zeros)
3. **Scaling Factor**: `α/r` controls the contribution of LoRA updates

### Parameter Efficiency

For a layer with dimensions `d × d`:
- **Standard Fine-tuning**: `d²` trainable parameters
- **LoRA**: `2 × d × r` trainable parameters (where r << d)
- **Memory Savings**: ~99% reduction with r=8 for d=4096

### Quantization Details

Uses BitsAndBytes `Linear4bit` with:
- **Compute dtype**: `bfloat16` for stable training
- **Storage**: 4-bit NormalFloat (NF4) format
- **Statistics**: Uncompressed for faster dequantization

## Configuration Parameters

### QLoRA Layer Parameters

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `alpha` | LoRA scaling factor | 4-32 (keep alpha/r ≈ 2) |
| `r` | LoRA rank | 2-64 (lower = more efficient) |
| `require_lora` | Enable LoRA adapters | `True` for fine-tuning |
| `use_qlora` | Use 4-bit quantization | `True` for memory efficiency |
| `quantize_weight` | Freeze base weights | `True` (standard practice) |

### Training Hyperparameters

```python
# Recommended starting point
learning_rate = 2e-4      # Higher LR due to limited trainable params
batch_size = 4            # Adjust based on VRAM
gradient_accumulation = 4 # Effective batch size = 16
alpha =32                 # Match to your rank
r = 16                     # Start small, increase if needed scale = alpha / r where recommended scale ~2 to 4
```

## Target Layers

Common layer patterns to quantize:

```python
# For LLaMA/Mistral models
target_layers = ["gate_proj", "up_proj", "down_proj"]  # MLP layers

# For attention layers
target_layers = ["q_proj", "k_proj", "v_proj", "o_proj"]

# For all linear layers (most aggressive)
target_layers = ["gate_proj", "up_proj", "down_proj", 
                 "q_proj", "k_proj", "v_proj", "o_proj"]
```

## Memory Comparison

Example for TinyLlama-1.1B:

| Method | Trainable Params | VRAM Usage | Training Speed |
|--------|------------------|------------|----------------|
| Full Fine-tuning | 1.1B (100%) | ~16 GB | 1x |
| LoRA (r=8) | ~4.2M (0.38%) | ~8 GB | 1.2x |
| QLoRA (r=8) | ~4.2M (0.38%) | ~4 GB | 1.1x |

## Verifying Your Setup

Check trainable parameters:

```python
def print_trainable(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} ({100*trainable/total:.3f}%)")
    print(f"Total: {total:,}")

print_trainable(model)
```

Expected output for TinyLlama with r=2:
```
✅ Trainable Params: 1,048,576 (0.095%)
Total: 1,100,048,384
```

## Implementation Details

### Weight Initialization

Following the LoRA paper:
- **W_B** initialized to zeros → LoRA starts as identity (no change initially)
- **W_A** initialized with Kaiming uniform → stable gradient flow

### Forward Pass Logic

```python
def forward(self, x):
    # Quantized base layer output
    base_output = self.quant_nn(x)
    
    if self.require_lora:
        # Cast adapters to input dtype
        W_a = self.W_a.to(x.dtype)
        W_b = self.W_b.to(x.dtype)
        
        # Apply LoRA: (x @ W_A) @ W_B scaled by α/r
        lora_output = self.scale * ((x @ W_a) @ W_b)
        
        return base_output + lora_output
    
    return base_output
```

### Recursive Quantization

The `quantize_model` function recursively traverses the model tree:
1. Finds all `nn.Linear` layers matching target names
2. Replaces them with `Qlora_4_bit` instances
3. Copies pre-trained weights to the quantized layer
4. Marks layers to prevent double-quantization

## Troubleshooting

### Common Issues

**"CUDA out of memory"**
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Use a smaller rank `r`

**"Loss not decreasing"**
- Increase `alpha` (try alpha/r = 2)
- Increase learning rate (try 2e-4 to 1e-3)
- Check that target layers are correct


**"No trainable parameters"**
- Verify `require_lora=True`
- Check target layer names match your model architecture

## Advanced Usage

### Using Standard LoRA (No Quantization)

```python
quantize_model(
    module=model,
    mlp_layers=target_layers,
    alpha=16,
    r=8,
    require_lora=True,
    use_qlora=False  # ← Disable 4-bit quantization
)
```

### Inference Only (No LoRA)

```python
quantize_model(
    module=model,
    mlp_layers=target_layers,
    alpha=4,
    r=2,
    require_lora=False,  # ← Disable LoRA adapters
    use_qlora=True       # Just quantize for inference
)
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- BitsAndBytes 0.41.0+
- Transformers 4.30+
- CUDA-compatible NVIDIA GPU (for training)

## References

- [QLoRA Paper](https://arxiv.org/abs/2305.14314) - Dettmers et al., 2023
- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Hu et al., 2021
- [BitsAndBytes Documentation](https://github.com/TimDettmers/bitsandbytes)

## Contributing

Contributions welcome! Areas for improvement:
- Support for 8-bit quantization
- Integration with more model architectures
- Advanced LoRA variants (AdaLoRA, QLoRA+)
- Comprehensive benchmarking suite


**Note**: This is an educational implementation demonstrating QLoRA internals. For production use, consider the official PEFT library which includes additional optimizations.