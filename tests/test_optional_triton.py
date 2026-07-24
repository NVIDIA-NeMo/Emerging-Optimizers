# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import subprocess
import sys
from pathlib import Path

import torch
from absl import flags, logging
from absl.testing import absltest


flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to run tests on")
flags.DEFINE_integer("seed", None, "Random seed for reproducible tests")
FLAGS = flags.FLAGS


def setUpModule() -> None:
    if FLAGS.seed is not None:
        logging.info("Setting random seed to %d", FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FLAGS.seed)


_NO_TRITON_SCRIPT = """
import importlib.abc
import sys

import torch


class TritonImportBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "triton" or fullname.startswith("triton."):
            raise ModuleNotFoundError("No module named 'triton'", name="triton")
        return None


sys.meta_path.insert(0, TritonImportBlocker())

from emerging_optimizers import triton_kernels
from emerging_optimizers.orthogonalized_optimizers import Muon


assert triton_kernels.HAS_TRITON_340 is False

param = torch.nn.Parameter(torch.randn(4, 4, device=sys.argv[1]))
initial_param = param.detach().clone()
param.grad = torch.randn_like(param)
Muon([param], use_syrk=True).step()

assert not torch.equal(initial_param, param)
assert torch.isfinite(param).all()
"""


class OptionalTritonTest(absltest.TestCase):
    def test_muon_without_triton(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", _NO_TRITON_SCRIPT, FLAGS.device],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            check=False,
            text=True,
        )

        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}",
        )


if __name__ == "__main__":
    absltest.main()
