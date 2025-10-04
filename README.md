# Emerging Optimizers

## Overview

Emerging Optimizers is a research project focused on understanding and optimizing the algorithmic behavior of emerging optimizers (including Shampoo, SOAP, Muon, and others) and their implications to performance of GPU systems in LLM training.

> ⚠️ Note: Emerging-Optimizers is under active development. All APIs are experimental and subject to change. New features, improvements, and documentation updates are released regularly. Your feedback and contributions are welcome, and we encourage you to follow along as new updates roll out.

## Background

### What are Emerging Optimizers?

Emerging optimizers represent a class of novel optimization algorithms that go beyond traditional first-order methods like Adam or SGD. These include optimizers that use matrix-based (non-diagonal) preconditioning, orthogonalization techniques, and other innovative approaches to achieve faster convergence and improved training efficiency.

Examples include Shampoo, which uses Kronecker-factored preconditioning ([arXiv:1802.09568](https://arxiv.org/abs/1802.09568)), and Muon, which uses Newton-Schulz orthogonalization ([arXiv:2502.16982](https://arxiv.org/abs/2502.16982)).

### Why They Matter

Emerging optimizers have demonstrated significant practical impact in large-scale language model training. Most notably, **Muon was used to train the Kimi K2 model** ([arXiv:2507.20534](https://arxiv.org/abs/2507.20534)), showcasing the effectiveness of these novel approaches at scale. These optimizers can:

- Achieve faster convergence, reducing the number of training steps required
- Improve final model quality through better conditioning of the optimization landscape
- Enable more efficient hyperparameter tuning due to reduced sensitivity to learning rates

### Optimizers Included

This project currently includes the following optimizers:

- **Shampoo**: Uses Kronecker-factored preconditioning ([arXiv:1802.09568](https://arxiv.org/abs/1802.09568))
- **SOAP (Shampoo with Adam in the Preconditioner)**: Combines Shampoo's preconditioning with Adam-style momentum ([arXiv:2409.11321](https://arxiv.org/abs/2409.11321))
- **Muon**: Uses Newton-Schulz orthogonalization for improved stability and convergence in large-scale training ([arXiv:2502.16982](https://arxiv.org/abs/2502.16982))


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

Muon (MomentUm Orthogonalized by Newton-schulz) uses orthogonalization for 2D parameters.

For a simple usage example, see [`tests/test_orthogonalized_optimizer.py::MuonTest`](tests/test_orthogonalized_optimizer.py).

### Integration with Megatron Core

Integration with Megatron Core is in progress. See the [integration PR](https://github.com/NVIDIA/Megatron-LM/pull/1813) that demonstrates usage with Dense and MoE models.

## Benchmarks

Coming soon.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
