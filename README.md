<div align="center">

# Emerging Optimizers

</div>

<div align="center">

<!-- Get the codecov badge with a token direct from https://app.codecov.io/gh/NVIDIA-NeMo -->
[![codecov](https://codecov.io/gh/NVIDIA-NeMo/Emerging-Optimizers/graph/badge.svg?token=IQ6U7IFYN0)](https://codecov.io/gh/NVIDIA-NeMo/Emerging-Optimizers)
[![CICD NeMo](https://github.com/NVIDIA-NeMo/Emerging-Optimizers/actions/workflows/cicd-main.yml/badge.svg?branch=main)](https://github.com/NVIDIA-NeMo/Emerging-Optimizers/actions/workflows/cicd-main.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
![GitHub Repo stars](https://img.shields.io/github/stars/NVIDIA-NeMo/Emerging-Optimizers)
[![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://docs.nvidia.com/nemo/emerging-optimizers/latest/index.html)

</div>

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

## Installation

### Prerequisites

- Python 3.10 or higher, 3.12 is recommended
- PyTorch 2.0 or higher

### Install from Source

```bash
git clone https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git
cd Emerging-Optimizers
pip install .
```

## Usage

### Example

Refer to tests for usage of different optimizers, e.g.  [`tests/test_orthogonalized_optimizer.py::MuonTest`](tests/test_orthogonalized_optimizer.py).

### Integration with Megatron Core

Integration with Megatron Core is available in **dev** branch, e.g. [muon.py](https://github.com/NVIDIA/Megatron-LM/blob/dev/megatron/core/optimizer/muon.py)

## Benchmarks

Coming soon.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
