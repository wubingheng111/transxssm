# TransXSSM: A Hybrid Transformer-Mamba Architecture Model

[![Paper](https://img.shields.io/badge/Paper-NIPS%202025%20Submission-red)](https://neurips.cc/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange)](https://pytorch.org/)

> **📝 NIPS 2025 Submission**  
> This repository contains the official implementation of **TransXSSM**, submitted to the 37th Conference on Neural Information Processing Systems (NIPS 2025). The paper introduces a novel hybrid architecture that combines the efficiency of State Space Models with the expressiveness of Transformer attention mechanisms.

## Abstract

TransXSSM is an innovative hybrid neural network architecture that seamlessly integrates Transformer's attention mechanisms with Mamba's State Space Model (SSM) technology. The model aims to provide powerful sequence modeling capabilities while maintaining computational efficiency through strategic layer composition.

## Key Features

### 🏗️ Hybrid Architecture Design
- **State Space Decoder (SSD) Layers**: Mamba-based efficient state space models providing linear complexity sequence processing
- **Attention Layers**: Traditional multi-head self-attention mechanisms enhancing model expressiveness
- **Flexible Layer Configuration**: Configurable periodic layer arrangement (default: one attention layer every 8 layers)

### ⚡ Performance Optimizations
- **Dynamic Caching Mechanism**: Supports `HybridSSDAttnDynamicCache` with optimized caching strategies for different layer types
- **RoPE Position Encoding**: Rotary Position Embeddings with dynamic scaling support
- **Efficient Inference**: Optimized Triton kernel support providing fast inference paths
- **Gradient Checkpointing**: Memory-efficient training support

### 🔧 Technical Features
- **Grouped Query Attention**: Supports MHA, MQA, and GQA configurations
- **Learnable Residual Connections**: Parameterized residual weights
- **RMSNorm Normalization**: Efficient layer normalization implementation
- **SiLU Activation**: Default Swish activation function

## Model Architecture

```
TransXSSM Model = Embedding + N×[SSD Layer|Attention Layer] + Output

Where:
- SSD Layer: State Space Decoder Layer (efficient sequence modeling)
- Attention Layer: Self-Attention Decoder Layer (enhanced expressiveness)
- Layer types controlled by attn_layer_period and attn_layer_offset
```

## Installation

```bash
# Core dependencies
pip install torch transformers

# Mamba SSM support (optional, for acceleration)
pip install mamba-ssm
```

## Quick Start

### Basic Usage

```python
import torch
from configuration_transxssm import TransXSSMConfig
from modeling_transxssm import TransXSSMForCausalLM

# Create configuration
config = TransXSSMConfig(
    vocab_size=32768,
    hidden_size=1024,
    num_hidden_layers=32,
    num_attention_heads=8,
    attn_layer_period=8,  # One attention layer every 8 layers
    attn_layer_offset=7   # First attention layer at position 7
)

# Initialize model
model = TransXSSMForCausalLM(config)

# Inference example
input_ids = torch.randint(0, config.vocab_size, (1, 128))
outputs = model(input_ids)
logits = outputs.logits
```

### Cache Usage

```python
from modeling_transxssm import HybridSSDAttnDynamicCache

# Create hybrid cache
cache = HybridSSDAttnDynamicCache(
    config, 
    batch_size=1,
    layer_type=config.layers_type
)

# Use cache for generation
outputs = model(input_ids, past_key_values=cache, use_cache=True)
```

## Configuration Parameters

### Core Parameters
- `vocab_size`: Vocabulary size (default: 32768)
- `hidden_size`: Hidden dimension (default: 1024)
- `num_hidden_layers`: Number of hidden layers (default: 32)
- `num_attention_heads`: Number of attention heads (default: 8)

### Hybrid Architecture Parameters
- `attn_layer_period`: Attention layer interval (default: 8)
- `attn_layer_offset`: First attention layer position (default: 7)
- `ssd_chunk_size`: SSD chunk size (default: 256)

### Position Encoding Parameters
- `max_position_embeddings`: Maximum position encoding length (default: 2048)
- `rope_theta`: RoPE base period (default: 10000.0)
- `rope_scaling`: RoPE scaling configuration (supports dynamic scaling)

## Project Structure

```
transxssm/
├── modeling_transxssm.py      # Main model implementation
├── configuration_transxssm.py # Configuration class definition
└── README.md                  # Project documentation
```

## Model Components

### TransXSSMStateSpace
State space module implementing Mamba's core SSM mechanism:
- Selective state update support
- Optimized chunk scan algorithms
- RoPE position encoding integration

### TransXSSMSelfAttention  
Multi-head self-attention module:
- Grouped Query Attention (GQA) support
- Attention dropout
- KV cache optimization

### HybridSSDAttnDynamicCache
Hybrid dynamic caching system:
- KV cache for attention layers
- State cache for SSD layers
- Automatic cache management and updates

## Performance Characteristics

1. **Computational Efficiency**: SSD layers provide O(n) complexity, attention layers provide O(n²) expressiveness
2. **Memory Optimization**: Hybrid caching mechanism reduces memory footprint
3. **Parallelism**: Supports gradient checkpointing and distributed training
4. **Scalability**: Flexible layer configuration adapts to different task requirements

## Applications

- Long sequence modeling tasks
- Causal language modeling
- Sequence classification tasks
- NLP applications requiring efficiency-performance balance

## Technical Details

### Hybrid Architecture Advantages
- **SSD Layers**: More efficient for long-range dependencies
- **Attention Layers**: Maintain strong modeling capabilities
- **Synergistic Design**: Two mechanisms complement each other for optimal performance

### Caching Strategy
- Attention layers use traditional key-value caching
- SSD layers use state caching
- Unified cache interface simplifies usage

## Citation

If you find this work useful for your research, please consider citing our NIPS 2025 submission:

```bibtex
@article{transxssm2025,
  title={TransXSSM: A Hybrid Transformer-Mamba Architecture for Efficient Sequence Modeling},
  author={[Authors]},
  journal={Advances in Neural Information Processing Systems},
  year={2025},
  note={Under review}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions to improve this project. Please feel free to submit issues and pull requests.

## Acknowledgments

We thank the open-source community for the foundational work on Transformers and Mamba architectures that made this research possible.

---

*TransXSSM: Bridging efficiency and expressiveness in neural sequence modeling*

