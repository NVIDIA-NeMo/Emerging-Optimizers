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
import torch
from emerging_optimizers.soap.soap import SOAP
import numpy as np
import random

config = {
    "lr": 0.001,
    "weight_decay": 0.01,
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "eps": 1e-8,
    "precondition_frequency": 1,
    "shampoo_beta": 0.95,
    "precondition_1d": False,
    "adam_warmup_steps": 1,
    "fp32_matmul_prec": "highest",
    "use_adaptive_criteria": False,
    "trace_normalization": False,
    "power_iter_steps": 1,
}


def main() -> None:
    # seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Define the size of the random matrix (parameter size).
    rows = 5
    cols = 3
    matrix_shape = (rows, cols)

    # Create a random matrix as a torch Parameter.
    param = torch.nn.Parameter(torch.randn(matrix_shape, device="cuda"))
    print(f"Param is on device: {param.device}")
    # Instantiate the custom SOAP optimizer with the random parameter.
    optimizer = SOAP(
        [param],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        betas=(config["adam_beta1"], config["adam_beta2"]),
        eps=config["eps"],
        precondition_frequency=config["precondition_frequency"],
        shampoo_beta=config["shampoo_beta"],
        precondition_1d=config["precondition_1d"],
        adam_warmup_steps=config["adam_warmup_steps"],
        trace_normalization=config["trace_normalization"],
        fp32_matmul_prec=config["fp32_matmul_prec"],
        use_adaptive_criteria=config["use_adaptive_criteria"],
        power_iter_steps=config["power_iter_steps"],
    )

    # Number of time steps (iterations) to simulate.
    time_steps = 11

    print("Initial parameter values:")
    print(param.data)
    print("---------------------------")

    # Simulate a time series of random gradients.
    for t in range(time_steps):
        # Simulate a random gradient matrix.
        random_gradient = torch.randn(matrix_shape, device=param.device)

        # In a normal training loop, backward() would populate .grad.
        # Here, we manually assign the gradient.
        param.grad = random_gradient

        # Run the optimizer step.
        optimizer.step()

        print("After time step", t + 1)
        print(param.data)
        print("----------")


if __name__ == "__main__":
    print("Starting SOAP tests....")
    main()
