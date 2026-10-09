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
"""Optimizer selection and disjoint parameter ownership for upstream GPT."""

import importlib
import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, override

import torch
from absl import logging

from emerging_optimizers import registry


if TYPE_CHECKING:
    from typing import overload


SCALAR_OPTIMIZERS = {"adamw", "lion", "signum", "laprop", "sim_ademamix"}
EXCLUDED_OPTIMIZERS = {"tp_rekls", "stacked_soap"}


def optimizer_classes() -> dict[str, type[torch.optim.Optimizer]]:
    """Register supported families, including the unregistered contributed Muown."""
    for module in (
        "orthogonalized_optimizers",
        "scalar_optimizers",
        "legacy_soap",
        "shampoo",
        "psgd",
        "riemannian_optimizers",
        "riemannian_optimizers.isospectral",
    ):
        importlib.import_module(f"emerging_optimizers.{module}")
    from emerging_optimizers.contrib.muown import Muown

    classes = {name: registry.get_optimizer_cls(name) for name in registry.get_optimizer_name_list()}
    return {name: cls for name, cls in classes.items() if name not in EXCLUDED_OPTIMIZERS} | {
        "adamw": torch.optim.AdamW,
        "muown": Muown,
    }


class OptimizerGroup(torch.optim.Optimizer):
    """Expose child optimizers as one optimizer, including one AMP overflow decision."""

    def __init__(self, optimizers: list[torch.optim.Optimizer]) -> None:
        self.optimizers = optimizers
        super().__init__([group for opt in optimizers for group in opt.param_groups], {})

    if TYPE_CHECKING:

        @overload
        def step(self, closure: None = ...) -> None: ...

        @overload
        def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    @override
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Apply all child updates; GradScaler checks all parameter groups first."""
        if closure is not None:
            raise ValueError("closure is not supported")
        for optimizer in self.optimizers:
            optimizer.step()
        return None

    @override
    def state_dict(self) -> dict[str, Any]:
        """Preserve each optimizer's own serialization format."""
        return {"optimizers": [optimizer.state_dict() for optimizer in self.optimizers]}

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore children and refresh group references replaced during loading."""
        states = state_dict["optimizers"]
        if len(states) != len(self.optimizers):
            raise ValueError("Checkpoint optimizer layout does not match the configured optimizer.")
        for optimizer, state in zip(self.optimizers, states, strict=True):
            optimizer.load_state_dict(state)
        self.param_groups = [group for opt in self.optimizers for group in opt.param_groups]


def decay_groups(params: list[torch.nn.Parameter], weight_decay: float) -> list[dict[str, Any]]:
    """Apply weight decay to matrices, leaving biases and normalization scales alone."""
    return [
        {"params": selected, "weight_decay": decay}
        for selected, decay in (
            ([p for p in params if p.ndim >= 2], weight_decay),
            ([p for p in params if p.ndim < 2], 0.0),
        )
        if selected
    ]


def build_optimizer(
    model: torch.nn.Module,
    name: str,
    learning_rate: float,
    adamw_learning_rate: float,
    weight_decay: float,
    optimizer_kwargs: dict[str, Any] | None = None,
    adamw_betas: tuple[float, float] = (0.9, 0.95),
    *,
    initialize: bool = True,
) -> OptimizerGroup:
    """Use scalar optimizers globally or pair a hidden-matrix optimizer with AdamW."""
    classes = optimizer_classes()
    if name not in classes:
        raise ValueError(f"Unknown optimizer {name!r}; choose from {sorted(classes)}")
    cls = classes[name]
    kwargs = dict(optimizer_kwargs or {})
    if {"params", "lr", "weight_decay"} & kwargs.keys():
        raise ValueError("Set params, lr, and weight_decay through the model and dedicated flags.")
    signature = inspect.signature(cls)
    allowed = set(signature.parameters)
    if name == "muon_hyperball":
        allowed |= set(inspect.signature(classes["muon"]).parameters)
        allowed.discard("weight_update_hook")
    unknown = kwargs.keys() - allowed
    if unknown:
        raise ValueError(f"Unsupported options for {name}: {sorted(unknown)}; valid options: {sorted(allowed)}")
    kwargs["lr"] = learning_rate
    if "weight_decay" in allowed:
        kwargs["weight_decay"] = weight_decay
    if name == "adaptive_muon":
        kwargs.setdefault("momentum", 0.95)
    if name == "adamw":
        kwargs.setdefault("betas", adamw_betas)

    params = [p for p in model.parameters() if p.requires_grad]
    if name in SCALAR_OPTIMIZERS:
        children = [cls(decay_groups(params, weight_decay), **kwargs)]
    else:
        # Collect by identity: the LM head and token embedding normally share one Parameter.
        auxiliary_ids = {
            id(p) for module in model.modules() if isinstance(module, torch.nn.Embedding) for p in module.parameters()
        }
        auxiliary_ids.update(id(p) for p in model.get_submodule("lm_head").parameters())
        matrices = [p for p in params if p.ndim == 2 and id(p) not in auxiliary_ids]
        matrix_ids = {id(p) for p in matrices}
        auxiliary = [p for p in params if id(p) not in matrix_ids]
        if not matrices:
            raise ValueError("The model has no hidden matrices to optimize.")
        if name == "muon_hyperball":
            radius = kwargs.setdefault("hyperball_radius", 1.0)
            if radius <= 0:
                raise ValueError("hyperball_radius must be positive")
            if initialize:
                logging.info("Rescaling hidden matrices to MuonHyperball radius %s", radius)
                with torch.no_grad():
                    for p in matrices:
                        norm = p.norm()
                        if norm == 0:
                            raise ValueError("MuonHyperball cannot initialize a zero matrix.")
                        p.mul_(radius / norm)
        children = [cls(matrices, **kwargs)]
        if auxiliary:
            children.append(
                torch.optim.AdamW(decay_groups(auxiliary, weight_decay), lr=adamw_learning_rate, betas=adamw_betas)
            )
    optimizer = OptimizerGroup(children)
    owned = [id(p) for group in optimizer.param_groups for p in group["params"]]
    if len(owned) != len(set(owned)) or set(owned) != {id(p) for p in params}:
        raise ValueError("Every trainable parameter must belong to exactly one optimizer group.")
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    for child in children:
        count = sum(p.numel() for group in child.param_groups for p in group["params"])
        logging.info("%s owns %d parameters", type(child).__name__, count)
    return optimizer
