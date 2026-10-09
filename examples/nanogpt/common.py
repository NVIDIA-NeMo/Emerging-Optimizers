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
"""Shared upstream imports, data loading, and precision configuration."""

import importlib.util
import pickle
import sys
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import ModuleType
from typing import Any

import torch


def load_nanogpt(directory: str) -> ModuleType:
    """Import model.py from an upstream checkout without executing train.py."""
    path = Path(directory).expanduser().resolve() / "model.py"
    if not path.is_file():
        raise FileNotFoundError(f"No nanoGPT model at {path}. Set --nanogpt_dir to a karpathy/nanoGPT checkout.")
    name = "_emerging_upstream_nanogpt_model"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolves annotations through sys.modules.
    spec.loader.exec_module(module)
    return module


def autocast_context(device: torch.device, dtype: str) -> Any:
    """Keep weights in FP32 and optionally autocast forward/backward operations."""
    if dtype == "float32":
        return nullcontext()
    if device.type == "cpu" and dtype == "float16":
        raise ValueError("CPU float16 is unsupported; use float32 or bfloat16.")
    return torch.autocast(device_type=device.type, dtype=getattr(torch, dtype))


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Read a nanoGPT checkpoint on CPU, including upstream configuration metadata."""
    return torch.load(path, map_location="cpu", weights_only=False)


def load_model_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Accept both compiled upstream checkpoints and ordinary state dictionaries."""
    model.load_state_dict({key.removeprefix("_orig_mod."): value for key, value in state.items()})


class TokenData:
    """Memory-map upstream uint16 train.bin/val.bin without requiring NumPy."""

    def __init__(self, directory: str, block_size: int) -> None:
        self.directory = Path(directory)
        self.block_size = block_size
        self.meta: dict[str, Any] = {}
        meta_path = self.directory / "meta.pkl"
        if meta_path.exists():
            with meta_path.open("rb") as stream:
                self.meta = pickle.load(stream)
        self.tokens = {}
        if sys.byteorder != "little":
            raise ValueError("The memory-mapped uint16 dataset requires a little-endian host.")
        for split in ("train", "val"):
            path = self.directory / f"{split}.bin"
            size = path.stat().st_size
            if size % 2 or size // 2 <= block_size:
                raise ValueError(f"{path} must contain more than {block_size} uint16 tokens.")
            self.tokens[split] = torch.from_file(str(path), shared=False, size=size // 2, dtype=torch.int16)

    def batch(
        self, split: str, batch_size: int, device: torch.device, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw contiguous next-token batches using an independent CPU RNG."""
        data = self.tokens[split]
        starts = torch.randint(len(data) - self.block_size, (batch_size,), generator=generator)
        windows = torch.stack([data[i : i + self.block_size + 1] for i in starts.tolist()])
        windows = windows.long().bitwise_and_(0xFFFF)
        x, y = windows[:, :-1].contiguous(), windows[:, 1:].contiguous()
        if device.type == "cuda":
            x, y = x.pin_memory(), y.pin_memory()
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@contextmanager
def optimizer_rng(device: torch.device, seed: int) -> Iterator[None]:
    """Give stochastic updates the same RNG on each DDP rank without changing dropout RNG."""
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    )
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        yield
