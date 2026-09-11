```{eval-rst}
.. role:: hidden
    :class: hidden-section

emerging_optimizers.embedding_optimizers
=========================================

.. automodule:: emerging_optimizers.embedding_optimizers
.. currentmodule:: emerging_optimizers.embedding_optimizers

Sinkhorn balancing
~~~~~~~~~~~~~~~~~~

.. autofunction:: sinkhorn_balance
```

`sinkhorn_balance` transforms tall, dense token-by-feature updates such as those for embedding tables,
Engram tables, and language-model prediction heads. It alternates row and column L2 normalization, then
applies the $\sqrt{n}$ scale that converts unit row L2 norm to unit row-wise RMS. The function is stateless:
momentum, Nesterov blending, learning-rate correction, and the final weight update remain the optimizer's
responsibility.

Near-zero rows are masked before balancing. `num_steps` counts individual axis-normalization steps and
must be odd: one initial row normalization is followed by column/row pairs. The DeepSeek-V4.1 defaults are
`num_steps=11`, `zero_row_threshold=1e-3`, and `eps=1e-20`. The function expects the token or vocabulary
dimension to be first and at least as large as the hidden dimension. Inputs must use BF16, FP16, or FP32.
It does not modify its input, performs balancing in an FP32 workspace, and returns a new tensor in the
input dtype and on the input device. See the
[DeepSeek-V4.1 technical report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf)
for the algorithm and pre-training configuration.
