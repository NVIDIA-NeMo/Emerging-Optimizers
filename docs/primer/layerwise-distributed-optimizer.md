# Layer-wise distributed optimizer

The latest NeMo stack supports a new type of distributed optimizer designed for preconditioner based optimizers like Muon.

## Overview

### Element-wise distributed optimizer

In traditional Data Parallelism, every GPU keeps a full copy of the optimizer states and weights. An **element-wise** distributed optimizer breaks this redundancy by:

- **Partitioning states:** Instead of every GPU storing everything, the optimizer states are **evenly** sliced up across all available GPUs.
- **Reduce-Scatter gradien**t: A reduce-scatter is performed over all gradients and each GPU gets portion of gradients corresponding to the parameters it "owns".
- **Local Updates:** Each GPU only updates the specific portion of the model parameters it "owns."
- **All-Gather parameters:** After the update, GPUs communicate to ensure everyone has the updated version of the full model for the next forward pass.

A more advanced version breaks down operations and overlaps communication with computations, e.g. the [Megatron-Core version](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/optimizer/distrib_optimizer.py).

### Preconditioner based optimizers

There are many emerging optimizers that requires gradient of the entire layer to calculate update to each individual weight. For example, the popular Muon optimizer does:
$$
orth()
$$
If weights and optimizer states are evenly distributed among DP ranks, update can't be calculated based on the data available on each GPU. Addition communication will be needed to collect data for calculating the full update.

### Layer-wise sharding

In a layer-wise distributed optimizer, parameters of different layers are distributed to different DP ranks. Each GPU has full layers worth of parameters so that preconditioner can be calculated.

<img src="_img\layerwise.png" alt="layerwise" style="zoom:20%;" />

One change comes with layer-wise is variable size communication, e.g. each GPU now needs to collect different size of updated parameters from different GPUs, aka [all_gatherv](https://www.mpich.org/static/docs/v3.2/www3/MPI_Allgatherv.html). The full Megatron-Core integration can be found in [layer_wise_optimizer.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/optimizer/layer_wise_optimizer.py).

There are further optimizations possible for the layer-wise distributed optimizers as well as different parallel strategy preconditioner based optimizer in general. They'll be introduced in future documents.

### Toy example

### 

## Kimi-K2 with Muon

Measured on 256xB200.

| Optimizer                      | TFLOPs/s | Memory consumption |
| ------------------------------ | -------- | ------------------ |
| Element-wise distributed AdamW |          |                    |
| Layer-wise distributed Muon    |          |                    |
|                                |          |                    |

