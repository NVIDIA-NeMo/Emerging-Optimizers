# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Callable, override

import torch
from torch import distributed as dist
from torch import optim
from torch.optim.optimizer import ParamsT


if TYPE_CHECKING:
    from typing import overload

from emerging_optimizers import mixin as opt_mixin
from emerging_optimizers import registry, utils
from emerging_optimizers.scalar_optimizers import update_functions
from emerging_optimizers.soap import soap, soap_utils
from emerging_optimizers.utils import FP32MatmulPrecT, get_pg_rank, get_pg_size


__all__ = ["TpRekls"]


@torch.no_grad()  # type: ignore[misc]
def all_gather_grad_and_kronecker_factors_tp(
    kronecker_factor_list: list[torch.Tensor],
    grad: torch.Tensor,
    partition_dim: int,
    tp_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """All-gathers a sharded gradient and its kronecker factors across the tensor parallel group.

    This is a simple implementation to support tensor parallel. It assumes grad is sharded among tensor parallel domain
    with partition_dim indicating the dimension it was sharded. To save memory, kronecker factors are also sharded
    but always along dimension 0 to make gather operation easy.

    Gradient and kronecker factors are both all-gathered to all tensor parallel ranks and returned so the caller
    can pass them to ``update_kronecker_factors`` (and downstream eigenbasis computations) without further
    communication.

    Args:
        kronecker_factor_list: List of preconditioner matrices (L and R), each sharded along dimension 0 across
            the tensor parallel group.
        grad: Local shard of the gradient tensor, sharded along ``partition_dim`` across the tensor parallel group.
        partition_dim: Dimension along which ``grad`` is sharded across the tensor parallel group.
        tp_group: Tensor parallel process group used to all-gather ``grad`` and ``kronecker_factor_list``.

    Returns:
        full_grad: Full (un-sharded) gradient tensor on every rank.
        full_kronecker_factor_list: List of full (un-sharded) kronecker factor matrices ``[L, R]`` on every rank.
    """
    tp_size = get_pg_size(tp_group)

    grad_shards = [torch.empty_like(grad) for _ in range(tp_size)]
    dist.all_gather(grad_shards, grad, group=tp_group)
    full_grad = torch.cat(grad_shards, dim=partition_dim)

    full_kronecker_factor_list: list[torch.Tensor] = []
    for kronecker_factor in kronecker_factor_list:
        factor_shards = [torch.empty_like(kronecker_factor) for _ in range(tp_size)]
        dist.all_gather(factor_shards, kronecker_factor, group=tp_group)
        full_kronecker_factor_list.append(torch.cat(factor_shards, dim=0))

    return full_grad, full_kronecker_factor_list


@registry.register_optimizer("tp_rekls")
class TpRekls(opt_mixin.WeightDecayMixin, optim.Optimizer):
    """Tensor-parallel variant of :class:`~emerging_optimizers.soap.rekls.REKLS`.

    Reimplemented from scratch (not inheriting from :class:`~emerging_optimizers.soap.soap.SOAP`) so the
    tensor-parallel bookkeeping stays isolated. Eigenbases are not stored in optimizer state; they are
    recomputed via :func:`~emerging_optimizers.soap.soap_utils.get_eigenbasis_eigh` from the kronecker
    factors. Each step calls eigh twice — once on the pre-update L, R for the
    :func:`~emerging_optimizers.soap.soap.update_kronecker_factors_kl_shampoo` correction, and once on
    the post-update L, R for the gradient projection.

    State per parameter (one entry per rank):
      - ``step``
      - ``exp_avg``, ``exp_avg_sq``: full-size tensors duplicated across ``tp_group`` ranks. ``exp_avg``
        is rotated through the basis change between steps (project back via the pre-update eigenbasis,
        then forward via the post-update eigenbasis), matching SOAP's
        :func:`~emerging_optimizers.soap.soap.update_eigenbasis_and_exp_avgs`. ``exp_avg_sq`` is not
        rotated, matching SOAP's eigh path.
      - ``L``, ``R``: kronecker factor matrices, sharded along dimension 0 across ``tp_group``.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups.
        lr: Learning rate.
        betas: Inner Adam betas ``(b1, b2)``.
        shampoo_beta: Beta for the kronecker factor moving average.
        eps: Inner Adam epsilon.
        weight_decay: Weight decay coefficient.
        weight_decay_method: See :class:`~emerging_optimizers.mixin.WeightDecayMixin`.
        tp_group: Process group across which parameters and gradients are sharded.
        fp32_matmul_prec: Precision for the optimizer-state GEMM operations.

    Note:
        Sharding is configured per-parameter-group via ``partition_dim`` (an int in ``{0, 1}``,
        or ``None`` for replicated parameters). Mixed-layout models should use one group per
        distinct ``partition_dim``::

            optimizer = TpRekls([
                {"params": column_parallel_params, "partition_dim": 0},
                {"params": row_parallel_params, "partition_dim": 1},
                {"params": replicated_params, "partition_dim": None},
            ], lr=1e-3, tp_group=tp_group)

        Groups without ``partition_dim`` use the default (``None`` → replicated, plain non-TP REKLS
        step on each rank, no collectives, full-size ``L``/``R``).
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.95),
        shampoo_beta: float = 0.95,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        *,
        weight_decay_method: opt_mixin.WeightDecayT = "decoupled",
        tp_group: dist.ProcessGroup,
        fp32_matmul_prec: FP32MatmulPrecT = "high",
    ) -> None:
        self.tp_group = tp_group
        self.tp_size = get_pg_size(tp_group)
        self.tp_rank = get_pg_rank(tp_group)

        self.weight_decay_method = weight_decay_method
        self.fp32_matmul_prec = fp32_matmul_prec

        defaults = {
            "lr": lr,
            "betas": betas,
            "shampoo_beta": shampoo_beta,
            "eps": eps,
            "weight_decay": weight_decay,
            "partition_dim": None,
        }
        super().__init__(params, defaults)

    @staticmethod
    def _validate_partition_dim(partition_dim: int | None) -> int | None:
        if partition_dim is not None and partition_dim not in (0, 1):
            raise ValueError(f"partition_dim must be 0, 1, or None, got {partition_dim}")
        return partition_dim

    @torch.no_grad()  # type: ignore[misc]
    def _init_group(self, group: dict, skip_non_grad_params: bool = True) -> None:
        partition_dim = self._validate_partition_dim(group["partition_dim"])
        for p in group["params"]:
            if skip_non_grad_params and p.grad is None:
                continue
            if p.dim() != 2:
                raise TypeError("TpRekls is only supported for 2D tensors")
            state = self.state[p]
            if len(state) == 0:
                m, n = p.shape
                # Get full size of m, n if the parameter is tensor-parallel.
                if partition_dim == 0:
                    m *= self.tp_size
                elif partition_dim == 1:
                    n *= self.tp_size

                # Both dimensions must be divisible by tp_size for the L/R shards (each sharded
                # along dim 0) to gather back to the full square shape via torch.cat.
                if partition_dim is not None and (m % self.tp_size or n % self.tp_size):
                    raise ValueError(
                        f"TpRekls requires both dimensions to be divisible by tp_size={self.tp_size}; "
                        f"got full shape ({m}, {n}) for a parameter with partition_dim={partition_dim}."
                    )

                state["step"] = 0
                state["exp_avg"] = torch.zeros((m, n), dtype=torch.float32, device=p.device)
                state["exp_avg_sq"] = torch.zeros((m, n), dtype=torch.float32, device=p.device)
                # Match init_kronecker_factors in soap.py: default dtype (typically float32).
                # L, R are sharded along dim 0 only when the param is tensor-parallel.
                shard = self.tp_size if partition_dim is not None else 1
                state["L"] = torch.zeros((m // shard, m), device=p.device)
                state["R"] = torch.zeros((n // shard, n), device=p.device)

    if TYPE_CHECKING:

        @overload
        def step(self, closure: None = ...) -> None: ...

        @overload
        def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()  # type: ignore[misc]
    @override
    def step(self, closure: None = None) -> None:
        assert closure is None, "No support for closure"
        for group in self.param_groups:
            self._init_group(group)

        for group in self.param_groups:
            partition_dim = self._validate_partition_dim(group["partition_dim"])
            for p in group["params"]:
                if p.grad is None:
                    continue  # pragma: no cover

                local_grad = p.grad.to(torch.float32)
                state = self.state[p]
                curr_iter_1_based = state["step"] + 1

                # Apply weight decay before the gather so l2 mode propagates into full_grad.
                self._apply_weight_decay_inplace(p, local_grad, group["lr"], group["weight_decay"])

                if partition_dim is None:
                    # Replicated parameter: no all-gather, state is already full-size.
                    full_grad = local_grad
                    kronecker_factor_list = [state["L"], state["R"]]
                else:
                    full_grad, kronecker_factor_list = all_gather_grad_and_kronecker_factors_tp(
                        kronecker_factor_list=[state["L"], state["R"]],
                        grad=local_grad,
                        partition_dim=partition_dim,
                        tp_group=self.tp_group,
                    )

                # Apply shampoo beta bias correction.
                shampoo_beta = group["shampoo_beta"]
                shampoo_beta = 1 - (1 - shampoo_beta) / (1 - shampoo_beta**curr_iter_1_based)

                # KL-Shampoo correction needs the eigenbasis of the *pre-update* L, R; recompute it
                # via eigh since we do not persist eigenbases across steps.
                with utils.fp32_matmul_precision(self.fp32_matmul_prec):
                    pre_eigenbasis_list = soap_utils.get_eigenbasis_eigh(kronecker_factor_list)
                    soap.update_kronecker_factors_kl_shampoo(
                        kronecker_factor_list,
                        full_grad,
                        shampoo_beta=shampoo_beta,
                        eigenbasis_list=pre_eigenbasis_list,
                        eps=group["eps"],
                    )

                # Persist the updated local shard back into state — only needed for the TP path,
                # since the replicated path updated state["L"], state["R"] in place via the alias.
                if partition_dim is not None:
                    state["L"].copy_(kronecker_factor_list[0].chunk(self.tp_size, dim=0)[self.tp_rank])
                    state["R"].copy_(kronecker_factor_list[1].chunk(self.tp_size, dim=0)[self.tp_rank])

                with utils.fp32_matmul_precision(self.fp32_matmul_prec):
                    # Rotate exp_avg from the pre-update eigenbasis to the post-update eigenbasis,
                    # and recompute the post-update eigenbasis via eigh.
                    eigenbasis_list, state["exp_avg"], state["exp_avg_sq"] = soap.update_eigenbasis_and_exp_avgs(
                        kronecker_factor_list=kronecker_factor_list,
                        eigenbasis_list=pre_eigenbasis_list,
                        exp_avg_sq=state["exp_avg_sq"],
                        exp_avg=state["exp_avg"],
                        use_eigh=True,
                    )

                    full_grad_projected = soap.precondition(full_grad, eigenbasis_list, dims=[[0], [0]])

                    # No matmul inside adam update. Put it under fp32_matmul_precision for code simplicity.
                    full_adam_update = update_functions.calculate_laprop_update(
                        full_grad_projected,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        betas=group["betas"],
                        eps=group["eps"],
                        correct_bias=True,
                        step=curr_iter_1_based,
                    )

                    full_precond_update = soap.precondition(full_adam_update, eigenbasis_list, dims=[[0], [1]])

                if partition_dim is None:
                    p.add_(full_precond_update, alpha=-group["lr"])
                else:
                    local_precond_update = full_precond_update.chunk(self.tp_size, dim=partition_dim)[self.tp_rank]
                    p.add_(local_precond_update, alpha=-group["lr"])

                state["step"] += 1

        return None
