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

.. autofunction:: aurora_orthogonalize


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
```

`Aurora` uses the same optimizer loop and state as `Muon`, replacing only
`scaled_orthogonalize_fn`. For tall updates stored as `[out, in]`, it balances
output-neuron rows with a diagonal preconditioner around the existing
Newton–Schulz solver. Square and wide updates keep the Muon path. Norm reductions
use FP32; exact zero rows remain zero. The update scale is applied once after all
outer solves. `eps` controls the floor for squared row norms and defaults to `1e-14`.

```python
from emerging_optimizers.orthogonalized_optimizers import Aurora

optimizer = Aurora(
    matrix_parameters,  # 2D FP32 parameters; use AdamW separately for other parameters
    lr=3e-4,
    num_aurora_steps=2,
    aurora_beta=0.5,
    coefficient_type="polar_express",
    num_ns_steps=8,
)
```

`num_ns_steps` counts inner iterations **per solve**, so two Aurora steps double
the number of Newton–Schulz solves for tall matrices. The coefficient schedule,
precision, shape scaling, and optional SYRK backend remain configurable through
the inherited Muon options. These defaults preserve this library's Muon recipe;
they do not reproduce every hyperparameter of the released Aurora optimizer.
The stateless `aurora_orthogonalize` helper also supports independent batches of
matrices, but the optimizer follows Muon's 2D-parameter contract. Pass complete,
unsharded matrices to the helper.

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
