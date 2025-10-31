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
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from absl import logging
from torchvision import datasets, transforms

from emerging_optimizers.soap.soap import SOAP


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 1024)  # MNIST images are 28x28
        self.fc2 = nn.Linear(1024, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


config = {
    "lr": 0.001,
    "weight_decay": 0.02,
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "eps": 1e-8,
    "precondition_1d": True,  # Enable preconditioning for bias vectors
    "precondition_frequency": 1,  # Update preconditioner every step for testing
    "fp32_matmul_prec": "high",
    "qr_fp32_matmul_prec": "high",
    "use_adaptive_criteria": False,
    "power_iter_steps": 1,
}


def train_step(model, optimizer, data, target, device, batch_idx, max_batches):
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()

    # Log gradient norms before optimizer step
    logging.debug(f"\nBatch {batch_idx + 1}/{max_batches}")
    logging.debug(f"Loss: {loss.detach().cpu().item():.4f}")
    for name, param in model.named_parameters():
        if param.grad is not None:
            logging.debug(f"{name} grad norm: {param.grad.norm():.4f}")

    optimizer.step()
    return loss.item()


def main() -> None:
    # Set seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST("./data", train=True, download=True, transform=transform),
        batch_size=256,
        shuffle=True,
    )

    # Initialize models and move to device
    model_soap = SimpleNN().to(device)
    model_adamw = SimpleNN().to(device)

    # Initialize with same weights
    model_adamw.load_state_dict(model_soap.state_dict())

    # Initialize optimizers
    optimizer_soap = SOAP(
        model_soap.parameters(),
        lr=2.1 * config["lr"],
        weight_decay=config["weight_decay"],
        betas=(config["adam_beta1"], config["adam_beta2"]),
        eps=config["eps"],
        precondition_frequency=config["precondition_frequency"],
        precondition_1d=config["precondition_1d"],
        fp32_matmul_prec=config["fp32_matmul_prec"],
        qr_fp32_matmul_prec=config["qr_fp32_matmul_prec"],
        use_adaptive_criteria=config["use_adaptive_criteria"],
        power_iter_steps=config["power_iter_steps"],
    )

    optimizer_adamw = torch.optim.AdamW(
        model_adamw.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        betas=(config["adam_beta1"], config["adam_beta2"]),
        eps=config["eps"],
    )

    # Train for a few steps
    logging.info("\nStarting training comparison...")
    model_soap.train()
    model_adamw.train()
    max_batches = 100

    soap_losses = []
    adamw_losses = []

    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx >= max_batches:
            break

        data, target = data.to(device), target.to(device)

        # Train SOAP model
        logging.debug("\nSOAP Optimizer:")
        soap_loss = train_step(model_soap, optimizer_soap, data, target, device, batch_idx, max_batches)
        soap_losses.append(soap_loss)

        # Train AdamW model
        logging.debug("\nAdamW Optimizer:")
        adamw_loss = train_step(model_adamw, optimizer_adamw, data, target, device, batch_idx, max_batches)
        adamw_losses.append(adamw_loss)

    logging.info("\nTraining completed!")
    logging.info("\nLoss comparison:")
    logging.info("Batch\tSOAP Loss\tAdamW Loss")
    for i, (soap_loss, adamw_loss) in enumerate(zip(soap_losses, adamw_losses)):
        logging.info(f"{i + 1}\t{soap_loss:.4f}\t{adamw_loss:.4f}")

    # Assert that SOAP's final loss is lower than AdamW's
    assert soap_losses[-1] < adamw_losses[-1], (
        f"SOAP's final loss ({soap_losses[-1]:.4f}) should be lower than AdamW's loss ({adamw_losses[-1]:.4f})"
    )

    print("SOAP's final loss:", soap_losses[-1])
    print("AdamW's final loss:", adamw_losses[-1])


if __name__ == "__main__":
    logging.info("Starting MNIST optimizer comparison test...")
    main()
