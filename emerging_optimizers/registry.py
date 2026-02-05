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
from functools import partial
from inspect import signature
from typing import Any, Callable

from torch import optim


_OPTIMIZERS: dict[str, type[optim.Optimizer]] = {}


def register_optimizer(name: str) -> Callable[[type], type]:
    """Decorator to register an optimizer class in the registry."""

    def decorator(cls: type) -> type:
        _OPTIMIZERS[name.lower()] = cls
        return cls

    return decorator


def get_optimizer(name: str) -> type[optim.Optimizer]:
    """Returns the optimizer class from the registry."""
    optimizer = _OPTIMIZERS.get(name.lower())
    if optimizer is None:
        raise ValueError(f"Optimizer {name} not found in the registry.")
    return optimizer


def validate_optimizer_args(opt_cls: type, kwargs: dict[str, Any]) -> None:
    """Checks if kwargs are valid for the optimizer class signature."""
    sig = signature(opt_cls)

    supported_params = set[str](sig.parameters.keys())
    unknown_args = set[str](kwargs.keys()) - supported_params
    unknown_args.discard("params")
    if unknown_args:
        raise TypeError(
            f"Optimizer '{opt_cls.__name__}' does not accept arguments: {unknown_args}.\n"
            f"Valid options are: {list(supported_params)}"
        )


def get_configured_optimizer(name: str, **kwargs: Any) -> Callable[[], optim.Optimizer]:
    """Returns a callable that creates an optimizer with the given arguments."""
    opt_cls = get_optimizer(name)
    validate_optimizer_args(opt_cls, kwargs)

    return partial(opt_cls, **kwargs)
