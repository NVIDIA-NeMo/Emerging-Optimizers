```{eval-rst}
.. role:: hidden
    :class: hidden-section

emerging_optimizers.embedding_optimizers
=========================================

.. automodule:: emerging_optimizers.embedding_optimizers
.. currentmodule:: emerging_optimizers.embedding_optimizers

Sinkhorn-balanced updates
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Sinkhorn
    :members:
```

`Sinkhorn` is a momentum optimizer for tall, dense token-by-feature matrices such as embedding tables,
Engram tables, and language-model prediction heads. It maintains one EMA momentum buffer and applies a
Nesterov blend before alternating row and column L2 normalization. A final odd row-normalization pass and
the $\sqrt{n}$ scale produce approximately unit RMS along both matrix axes.

Near-zero rows are masked before balancing. `num_steps` must be odd so the last pass normalizes rows, and
the DeepSeek-V4.1 defaults are `momentum=0.95`, `num_steps=11`, `zero_row_threshold=1e-3`, `eps=1e-20`,
and `lr_correction=0.18`. The optimizer expects the token or vocabulary dimension to be the first and
larger matrix dimension. Parameters must use BF16, FP16, or FP32; FP64 parameters are rejected explicitly.
The balancing workspace uses FP32 for all supported parameter dtypes. Sparse PyTorch gradients are not
supported because column normalization requires the full dense update matrix. See the
[DeepSeek-V4.1 technical report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf)
for the algorithm and pre-training configuration.
