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
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, override


if TYPE_CHECKING:
    from typing import overload

import torch
from torch.optim.optimizer import ParamsT

from emerging_optimizers.mixin import WeightDecayMixin, WeightDecayT


__all__ = [
    "SingleMomentumOptimizer",
    "TwoMomentsOptimizer",
]


def _validate_common_hparams(
    *,
    lr: float | None = None,
    betas: tuple[float, ...] | None = None,
    eps: float | None = None,
    weight_decay: float | None = None,
) -> None:
    """Validates the hyperparameters shared by most scalar optimizers."""
    if lr is not None and lr < 0.0:
        raise ValueError(f"Invalid learning rate: {lr}")
    if betas is not None:
        for i, b in enumerate(betas):
            if not 0.0 <= b < 1.0:
                raise ValueError(f"Invalid beta at index {i}: {b}")
    if eps is not None and eps < 0.0:
        raise ValueError(f"Invalid epsilon value: {eps}")
    if weight_decay is not None and weight_decay < 0.0:
        raise ValueError(f"Invalid weight_decay value: {weight_decay}")


class _ScalarOptimizerBase(WeightDecayMixin, torch.optim.Optimizer):
    """Shared implementation for scalar optimizers grouped by state shape.

    Subclasses set ``state_keys`` as a ``ClassVar``. The base lazily allocates one
    zero-initialized buffer per name plus a per-parameter ``step`` counter, then
    dispatches each step to a constructor-supplied ``update_fn`` whose signature is
    ``update_fn(grad, *buffers, **kwargs) -> Tensor``.

    Hyperparameters forwarded into ``update_fn`` are selected from the parameter
    group via ``update_kwarg_names`` (a tuple of dict keys present in the
    ``defaults`` mapping). The per-parameter ``step`` is always forwarded as
    ``step=state["step"]``, so every update function must accept a ``step`` kwarg.

    Subclasses can additionally override :meth:`pre_step_inplace` /
    :meth:`post_step_inplace` to bracket the per-parameter update with custom
    logic (e.g. norm preservation).
    """

    state_keys: ClassVar[tuple[str, ...]]

    def __init__(
        self,
        params: ParamsT,
        defaults: dict[str, Any],
        *,
        update_fn: Callable[..., torch.Tensor],
        update_kwarg_names: tuple[str, ...],
        weight_decay_method: WeightDecayT = "decoupled",
    ) -> None:
        missing = set(update_kwarg_names) - set(defaults.keys())
        if missing:
            raise ValueError(
                f"update_kwarg_names {sorted(missing)} not present in defaults (keys: {sorted(defaults.keys())})"
            )
        self.update_fn = update_fn
        self.update_kwarg_names = update_kwarg_names
        self.weight_decay_method = weight_decay_method
        super().__init__(params, defaults)

    @torch.no_grad()
    def _init_group(
        self,
        group: dict,
        skip_non_grad_params: bool = True,
    ) -> None:
        """Performs lazy state initialization for parameters."""
        for p in group["params"]:
            if skip_non_grad_params and p.grad is None:
                continue
            state = self.state[p]
            if len(state) == 0:
                for key in self.state_keys:
                    state[key] = torch.zeros_like(p.data)
                state["step"] = 0

    def pre_step_inplace(self, p: torch.Tensor, group: dict) -> Any:
        """Hook called before weight decay and the update. Return value is forwarded to ``post_step_inplace``."""
        return None

    def post_step_inplace(self, p: torch.Tensor, group: dict, ctx: Any) -> None:
        """Hook called after the update. Receives the value returned by ``pre_step_inplace``."""
        return None

    if TYPE_CHECKING:

        @overload
        def step(self, closure: None = ...) -> None: ...

        @overload
        def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()  # type: ignore[misc]
    @override
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform a single optimization step.

        Note:
            When ``weight_decay_method="l2"``, ``p.grad`` is modified in-place
            (the L2 penalty ``weight_decay * p`` is added to the gradient).
            If you need the original gradient after this call, clone it beforehand.

        Args:
            closure: Unsupported; must be ``None``.
        """
        if closure is not None:
            raise ValueError("closure is not supported")

        for group in self.param_groups:
            self._init_group(group)

            lr = group["lr"]
            weight_decay = group["weight_decay"]
            update_kwargs = {key: group[key] for key in self.update_kwarg_names}

            for p in group["params"]:
                if p.grad is None:
                    continue  # pragma: no cover

                state = self.state[p]
                state["step"] += 1
                update_kwargs["step"] = state["step"]

                ctx = self.pre_step_inplace(p, group)
                self._apply_weight_decay_inplace(p.data, p.grad, lr, weight_decay)

                buffers = tuple(state[key] for key in self.state_keys)
                update = self.update_fn(p.grad, *buffers, **update_kwargs)
                p.data.add_(update, alpha=-lr)

                self.post_step_inplace(p, group, ctx)

        return None


class SingleMomentumOptimizer(_ScalarOptimizerBase):
    """Base for scalar optimizers tracking a single first-moment EMA buffer."""

    state_keys: ClassVar[tuple[str, ...]] = ("exp_avg",)


class TwoMomentsOptimizer(_ScalarOptimizerBase):
    """Base for Adam-style scalar optimizers tracking first + second moment buffers."""

    state_keys: ClassVar[tuple[str, ...]] = ("exp_avg", "exp_avg_sq")
