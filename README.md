# Emerging Optimizers

## Overview

Emerging Optimizers is a research project focused on understanding and optimizing the algorithmic behavior of Shampoo class optimizers (Shampoo, SOAP, Muon, etc.) and their implications to performance of GPU systems in LLM training.

> ⚠️ Note: Emerging-Optimizers is under active development. All APIs are experimental and subject to change. New features, improvements, and documentation updates are released regularly. Your feedback and contributions are welcome, and we encourage you to follow along as new updates roll out.

## Background

### What are Shampoo Optimizers?

Shampoo class optimizers are a family of second-order optimization algorithms that use preconditioned gradient descent to achieve faster convergence compared to traditional first-order methods like Adam or SGD. Unlike these conventional optimizers that treat each parameter independently, Shampoo optimizers leverage the full or block-wise structure of the gradient's second moment statistics through matrix preconditioning.

**Reference:** [Shampoo: Preconditioned Stochastic Tensor Optimization](https://arxiv.org/abs/1802.09568)

### Why They Matter

Shampoo optimizers have demonstrated significant practical impact in large-scale language model training. Most notably, they were used to train the **Kimi K2 model** ([arXiv:2507.20534](https://arxiv.org/abs/2507.20534)), showcasing their effectiveness at scale. These optimizers can:

- Achieve faster convergence, reducing the number of training steps required
- Improve final model quality through better conditioning of the optimization landscape
- Enable more efficient hyperparameter tuning due to reduced sensitivity to learning rates

### Performance Challenges

Despite their algorithmic advantages, Shampoo class optimizers present unique challenges for GPU systems:

- **Memory overhead**: Storing and updating preconditioner matrices requires significantly more memory than first-order methods
- **Computational complexity**: Computing matrix roots and inverses for preconditioning introduces computational bottlenecks
- **Communication costs**: In distributed training, synchronizing larger optimizer states can impact scalability

### Optimizers Included

This project focuses on the following Shampoo class optimizers:

- **Shampoo**: The foundational second-order optimizer using Kronecker-factored preconditioning ([arXiv:1802.09568](https://arxiv.org/abs/1802.09568))
- **SOAP (Shampoo with Adam in the Preconditioner)**: A variant that combines Shampoo's preconditioning with Adam-style momentum ([arXiv:2409.11321](https://arxiv.org/abs/2409.11321))
- **Muon**: A momentum-based variant designed for improved stability and convergence in large-scale training ([arXiv:2502.16982](https://arxiv.org/abs/2502.16982))


## Installation

### Prerequisites

- Python 3.12 or higher
- PyTorch 2.0 or higher

### Install from Source

```bash
git clone https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git
cd Emerging-Optimizers
pip install .
```

## Usage

### Muon Optimizer

Muon (MomentUm Orthogonalized by Newton-schulz) uses orthogonalization for 2D parameters:

```python
import torch
from emerging_optimizers.orthogonalized_optimizers import Muon

# Create model
model = YourModel()

# Separate parameters: use Muon for 2D weights, Adam for others
muon_params = [p for p in model.parameters() if p.ndim == 2]
adam_params = [p for p in model.parameters() if p.ndim != 2]

# Initialize optimizers
optimizer_muon = Muon(
    muon_params,
    lr=3e-4,
    momentum_beta=0.95,
    weight_decay=0.01,
)

optimizer_adam = torch.optim.AdamW(
    adam_params,
    lr=3e-4,
    weight_decay=0.01,
)

# Training loop
for batch in dataloader:
    for opt in [optimizer_muon, optimizer_adam]:
        opt.zero_grad()
    loss = model(batch)
    loss.backward()
    for opt in [optimizer_muon, optimizer_adam]:
        opt.step()
```

### Integration with Megatron Core

For large-scale training with Megatron Core, see the [integration example](https://github.com/NVIDIA/Megatron-LM/pull/1813) that demonstrates usage with Dense and MoE models.

## Benchmarks

Coming soon.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
