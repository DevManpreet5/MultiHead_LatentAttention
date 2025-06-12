# Multi-Head Latent Attention (MHLA)

A simple PyTorch implementation of Multi-Head Latent Attention inspired by DeepSeek, without RoPE positional encodings.

## Usage

```python
import torch
import torch.nn as nn

d_model = 512
n_heads = 8
kv_latent_dim = 128
batch_size = 2
seq_len = 16

model = ropeless(d_model=d_model, kv_latent_dim=kv_latent_dim, num_head=n_heads)

x = torch.randn(batch_size, seq_len, d_model)
output, kv_cache = model(x)
```

## Parameters

- `d_model`: Model dimension (input/output size)
- `kv_latent_dim`: Compressed latent dimension for keys/values
- `num_head`: Number of attention heads

## Architecture

1. Input → Latent compression (`W_dkv`)
2. Latent → Key/Value upsampling (`W_uk`, `W_uv`)
3. Query projection (`W_q`)
4. Multi-head attention with causal masking
5. Output projection (`W_o`)
