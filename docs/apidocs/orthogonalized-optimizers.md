```{eval-rst}  
.. role:: hidden
    :class: hidden-section

emerging_optimizers.orthogonalized_optimizers
=============================================

.. automodule:: emerging_optimizers.orthogonalized_optimizers
.. currentmodule:: emerging_optimizers.orthogonalized_optimizers

:hidden:`OrthogonalizedOptimizer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: OrthogonalizedOptimizer
    :members:

:hidden:`Muon`
~~~~~~~~~~~~~~~

.. autoclass:: Muon
    :members:


:hidden:`Aurora`
~~~~~~~~~~~~~~~

.. autoclass:: Aurora
    :members:


:hidden:`Scion`
~~~~~~~~~~~~~~~

.. autoclass:: Scion
    :members:

:hidden:`MOP`
~~~~~~~~~~~~~~~

.. autoclass:: MOP
    :members:

:hidden:`MuonHyperball`
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MuonHyperball
    :members:


:hidden:`PolarGrad`
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: PolarGrad
    :members:

.. autofunction:: one_sided_polargrad_orth_fn

.. autofunction:: left_polargrad_orth_fn

.. autofunction:: right_polargrad_orth_fn


:hidden:`AdaptiveMuon`
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AdaptiveMuon
    :members:


:hidden:`Spel`
~~~~~~~~~~~~~~~

.. currentmodule:: emerging_optimizers.orthogonalized_optimizers.spel

.. autoclass:: Spel
    :members:

.. autofunction:: get_spel_scale_factor


:hidden:`Newton-Schulz`
~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: emerging_optimizers.orthogonalized_optimizers.muon_utils
.. currentmodule:: emerging_optimizers.orthogonalized_optimizers.muon_utils

.. autofunction:: newton_schulz
.. autofunction:: newton_schulz_step
.. autofunction:: newton_schulz_tp

:hidden:`Sinkhorn balancing`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: emerging_optimizers.orthogonalized_optimizers.sinkhorn_utils
.. currentmodule:: emerging_optimizers.orthogonalized_optimizers.sinkhorn_utils

.. autofunction:: sinkhorn_balance

.. autofunction:: sinkhorn_balance_row_sharded
```

`Aurora` subclasses `OrthogonalizedOptimizer` directly and supplies its own
`scaled_orthogonalize_fn`, sharing the base optimizer loop and state with `Muon`.
For tall updates stored as `[out, in]`, it balances
output-neuron rows with a diagonal preconditioner around the existing
Newton–Schulz solver. Square and wide updates keep the Muon path. Norm reductions
use FP32; exact zero rows remain zero. The update scale is applied once after all
outer solves. `eps` defaults to `1e-6` and specifies a fraction of the matrix's
average row L2 norm: each balancing stage uses `eps * mean(row_norms)` as its norm
floor. This relative floor preserves the tall-matrix update under positive scaling
of the momentum. The scaled orthogonalization callback also accepts batches of
matrices, computing each matrix's floor independently; the base optimizer still
expects 2D parameters. All-zero matrices produce a finite zero update.

`sinkhorn_balance` transforms tall, dense token-by-feature updates such as those for embedding tables,
Engram tables, and language-model prediction heads. It alternates row and column L2 normalization, then
applies the $\sqrt{n}$ scale that converts unit row L2 norm to unit row-wise RMS. The function is stateless:
momentum, Nesterov blending, learning-rate correction, and the final weight update remain the optimizer's
responsibility.

Near-zero rows are masked before balancing. `num_normalization_steps` counts individual axis-normalization
steps and must be odd: one initial row normalization is followed by column/row pairs. The DeepSeek-V4.1
defaults are `num_normalization_steps=11`, `zero_row_threshold=1e-3`, and `eps=1e-20`. The function expects
the token or vocabulary dimension to be first and at least as large as the hidden dimension. Inputs must
use BF16, FP16, or FP32.
It does not modify its input, performs balancing in an FP32 workspace, and returns a new tensor in the
input dtype and on the input device. See the
[DeepSeek-V4.1 technical report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf)
for the algorithm and pre-training configuration.

### Row-sharded Sinkhorn balancing

`sinkhorn_balance_row_sharded` accepts a dense **local** tensor with shape
`(local_rows, num_features)`. All feature columns of each local row must be
present. The logical matrix must be nonempty and tall (`global_num_rows >=
num_features > 0`), but a local shard may be wide or empty. FP32 workspace
accumulation and output dtype/device preservation match the serial API. The
existing `sinkhorn_balance` signature and validation are unchanged; both entry
points share the same normalization loop.

```python
from emerging_optimizers.orthogonalized_optimizers import sinkhorn_balance_row_sharded

balanced = sinkhorn_balance_row_sharded(
    local_update,
    row_group=matrix_row_group,
    global_num_rows=table_num_rows,
    num_normalization_steps=11,
    zero_row_threshold=1e-3,
    eps=1e-20,
)
```

`row_group` must contain distinct, nonoverlapping row owners for **one logical
matrix**, not replicas. The caller must provide the same `global_num_rows`,
feature width, numerical settings, and call order on every group member, and
ensure the local row counts sum to `global_num_rows`. That count is supplied as
metadata rather than reduced every step. No partition offsets are needed since
row/column norms are invariant to row order. Padding is not part of the logical
matrix and must be removed by the caller. Plain local tensors, not DTensors with
unresolved placements, are the input contract.

Every member must call the function, including members with zero local rows.
Global mode reduces the initial sum of row L2 norms and each partial column
sum-of-squares, taking the square root and adding epsilon **after** reduction.
For 11 individual normalization steps this is one scalar and five feature-vector
SUM reductions, with no matrix all-gather. Uneven partitions use the global row
count for the masking threshold, not the average of rank-local means. Norms,
thresholds, and normalization are computed in FP32. Masked rows remain zero;
finite iteration count and masking do not guarantee unit RMS for every column.

Passing `row_group=None` means **no communication**, never the default WORLD
group. In global mode, None or a size-one group requires the full logical matrix
locally. An explicit group requires initialized `torch.distributed` and current
rank membership; the function does not initialize process groups. Group/backend
selection and cross-rank configuration validation remain the caller's
responsibility.

`use_shard_local_stats=True` selects an explicitly approximate mode. It ignores
`row_group` and uses the local shard's own masking threshold and column norms,
with no collectives even if a group was supplied. The same shape requirements
apply in both modes. Changing row partitions can change the resulting update;
local balancing is not equivalent to global balancing. The mode does not remove
the existing communication needed to finalize gradients or synchronize weights.

Momentum, Nesterov blending, per-table boundaries in fused storage,
learning-rate correction, final weight updates, and optimizer-state placement
remain the optimizer/integration layer's responsibility. Independent logical
hash tables should call this function separately; storage fusion does not
combine their statistics. FSDP integration is the caller's responsibility. This
API does not implement feature sharding, asynchronous scheduling, fused kernels,
or batching.
