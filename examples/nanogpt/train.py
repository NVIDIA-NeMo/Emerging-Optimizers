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
"""Train an imported nanoGPT model with absl flags and Emerging Optimizers."""

import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from absl import app, flags, logging
from common import (
    TokenData,
    autocast_context,
    load_checkpoint,
    load_model_state,
    load_nanogpt,
    optimizer_rng,
)
from optimizers import build_optimizer, optimizer_classes
from torch.nn.parallel import DistributedDataParallel


FLAGS = flags.FLAGS
_EXISTING_FLAGS = set(FLAGS)
flags.DEFINE_string("nanogpt_dir", "../nanoGPT", "Path to an upstream karpathy/nanoGPT checkout.")
flags.DEFINE_string(
    "data_dir", "data/shakespeare_char", "Directory containing train.bin, val.bin, and optional meta.pkl."
)
flags.DEFINE_string("out_dir", "out/nanogpt", "Checkpoint and metrics directory.")
flags.DEFINE_enum(
    "init_from", "scratch", ["scratch", "resume", "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"], "Initialization."
)
flags.DEFINE_string("checkpoint", None, "Checkpoint to resume; defaults to out_dir/ckpt.pt.")
flags.DEFINE_string("optimizer", "muon", "Optimizer name; see --list_optimizers.")
flags.DEFINE_bool("list_optimizers", False, "List supported optimizer names and exit.")
flags.DEFINE_string("optimizer_kwargs", "{}", 'JSON constructor options, e.g. {"momentum": 0.95, "num_ns_steps": 5}.')
flags.DEFINE_float("learning_rate", 3e-4, "Main optimizer peak learning rate.", lower_bound=0.0)
flags.DEFINE_float("adamw_learning_rate", 3e-4, "Auxiliary AdamW peak learning rate.", lower_bound=0.0)
flags.DEFINE_float(
    "weight_decay", 0.1, "Matrix weight decay; biases and normalization scales use zero.", lower_bound=0.0
)
flags.DEFINE_float("beta1", 0.9, "AdamW first moment beta.")
flags.DEFINE_float("beta2", 0.95, "AdamW second moment beta.")
flags.DEFINE_integer("n_layer", 6, "Transformer layers.", lower_bound=1)
flags.DEFINE_integer("n_head", 6, "Attention heads.", lower_bound=1)
flags.DEFINE_integer("n_embd", 384, "Embedding dimension.", lower_bound=1)
flags.DEFINE_integer("block_size", 256, "Context length; may crop a pretrained model.", lower_bound=1)
flags.DEFINE_integer("vocab_size", None, "Vocabulary size; otherwise metadata or GPT-2's padded 50304.", lower_bound=1)
flags.DEFINE_float("dropout", 0.0, "Model dropout.", lower_bound=0.0, upper_bound=1.0)
flags.DEFINE_bool("bias", False, "Use linear and LayerNorm biases.")
flags.DEFINE_integer("batch_size", 12, "Microbatch size per rank.", lower_bound=1)
flags.DEFINE_integer(
    "gradient_accumulation_steps", 1, "Global microbatches per update; divisible by world size.", lower_bound=1
)
flags.DEFINE_integer(
    "max_iters", 5000, "Total optimizer iterations, including previously completed iterations.", lower_bound=0
)
flags.DEFINE_float("grad_clip", 1.0, "Global gradient norm clip; zero disables clipping.", lower_bound=0.0)
flags.DEFINE_bool("decay_lr", True, "Use linear warmup and cosine learning-rate decay.")
flags.DEFINE_integer("warmup_iters", 100, "Learning-rate warmup iterations.", lower_bound=0)
flags.DEFINE_integer("lr_decay_iters", 5000, "Iteration at which cosine decay reaches its minimum.", lower_bound=1)
flags.DEFINE_float("min_lr", 3e-5, "Main optimizer minimum LR; auxiliary groups keep their LR ratio.", lower_bound=0.0)
flags.DEFINE_integer("eval_interval", 250, "Evaluate and consider checkpointing every N updates.", lower_bound=1)
flags.DEFINE_integer("eval_iters", 20, "Batches per split and rank during evaluation.", lower_bound=1)
flags.DEFINE_bool("eval_only", False, "Evaluate once and exit, including when resuming.")
flags.DEFINE_bool(
    "always_save_checkpoint", True, "Save at each evaluation; otherwise save improvements and the final state."
)
flags.DEFINE_integer("log_interval", 10, "Log training metrics every N updates.", lower_bound=1)
flags.DEFINE_string("device", "cuda", "cpu, cuda, or a CUDA device such as cuda:0.")
flags.DEFINE_enum(
    "dtype", "bfloat16", ["float32", "bfloat16", "float16"], "Forward/backward precision; parameters remain FP32."
)
flags.DEFINE_bool("compile", False, "Compile the model with torch.compile.")
flags.DEFINE_enum("backend", "nccl", ["nccl", "gloo"], "DDP process-group backend; use gloo for CPU.")
flags.DEFINE_integer("seed", 1337, "Random seed.")
flags.DEFINE_bool("wandb_log", False, "Enable optional Weights & Biases logging.")
flags.DEFINE_string("wandb_project", "nanogpt", "W&B project.")
flags.DEFINE_string("wandb_run_name", None, "W&B run name.")
_CONFIG_FLAGS = set(FLAGS) - _EXISTING_FLAGS


def lr_multiplier(step: int, config: dict[str, Any]) -> float:
    """Scale all groups together while preserving their configured LR ratios."""
    if not config["decay_lr"]:
        return 1.0
    if step < config["warmup_iters"]:
        return (step + 1) / (config["warmup_iters"] + 1)
    minimum = config["min_lr"] / config["learning_rate"] if config["learning_rate"] else 0.0
    if step >= config["lr_decay_iters"]:
        return minimum
    progress = (step - config["warmup_iters"]) / (config["lr_decay_iters"] - config["warmup_iters"])
    return minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * progress))


def train(config: dict[str, Any]) -> None:
    """Run single-device or torchrun DDP training, evaluation, and checkpointing."""
    ddp = "RANK" in os.environ
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(config["device"])
    if ddp and device.type == "cuda":
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    if device.type == "cuda":
        device = torch.device("cuda", device.index if device.index is not None else 0)
        torch.cuda.set_device(device)
    if ddp:
        dist.init_process_group(backend=config["backend"])
    try:
        _train(config, device, rank, world_size, ddp)
    finally:
        if ddp:
            dist.destroy_process_group()


def _train(config: dict[str, Any], device: torch.device, rank: int, world_size: int, ddp: bool) -> None:
    output = Path(config["out_dir"])
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    checkpoint = None
    if config["init_from"] == "resume":
        checkpoint = load_checkpoint(Path(config["checkpoint"] or output / "ckpt.pt"))
        if "emerging_optimizer" not in checkpoint:
            raise ValueError(
                "Resume needs an Emerging Optimizers checkpoint; upstream checkpoints can be used by sample.py."
            )
        # Build the data loader with the context length stored in the model checkpoint.
        config["block_size"] = checkpoint["model_args"]["block_size"]
        # Constructor choices and the schedule are part of the optimizer state, not restart overrides.
        for key in (
            "seed",
            "optimizer",
            "optimizer_kwargs",
            "learning_rate",
            "adamw_learning_rate",
            "weight_decay",
            "beta1",
            "beta2",
            "decay_lr",
            "warmup_iters",
            "lr_decay_iters",
            "min_lr",
        ):
            config[key] = checkpoint["config"][key]
    if config["gradient_accumulation_steps"] % world_size:
        raise ValueError("gradient_accumulation_steps must be divisible by the DDP world size.")
    accumulation = config["gradient_accumulation_steps"] // world_size
    if config["decay_lr"] and (
        config["lr_decay_iters"] <= config["warmup_iters"] or config["min_lr"] > config["learning_rate"]
    ):
        raise ValueError("Require lr_decay_iters > warmup_iters and min_lr <= learning_rate.")
    kwargs = json.loads(config["optimizer_kwargs"])
    if not isinstance(kwargs, dict):
        raise ValueError("optimizer_kwargs must be a JSON object.")

    torch.manual_seed(config["seed"])
    upstream = load_nanogpt(config["nanogpt_dir"])
    data = TokenData(config["data_dir"], config["block_size"])
    model_args = {key: config[key] for key in ("n_layer", "n_head", "n_embd", "block_size", "dropout", "bias")}
    model_args["vocab_size"] = config["vocab_size"] or data.meta.get("vocab_size", 50304)
    model: Any
    if checkpoint is not None:
        model_args = dict(checkpoint["model_args"])
        model = upstream.GPT(upstream.GPTConfig(**model_args))
        load_model_state(model, checkpoint["model"])
    elif config["init_from"] == "scratch":
        model = upstream.GPT(upstream.GPTConfig(**model_args))
    else:
        model = upstream.GPT.from_pretrained(config["init_from"], {"dropout": config["dropout"]})
        model_args = vars(model.config).copy()
    if config["block_size"] > model.config.block_size:
        raise ValueError("block_size cannot exceed the resumed/pretrained model's context length.")
    if config["block_size"] < model.config.block_size:
        if checkpoint is not None:
            raise ValueError(
                "Changing block_size on resume would invalidate optimizer state; keep the saved block_size."
            )
        model.crop_block_size(config["block_size"])
        model_args["block_size"] = config["block_size"]
    if data.meta.get("vocab_size", 0) > model.config.vocab_size:
        raise ValueError("Dataset vocabulary exceeds the model vocabulary.")
    model.to(device)
    optimizer = build_optimizer(
        model,
        config["optimizer"],
        config["learning_rate"],
        config["adamw_learning_rate"],
        config["weight_decay"],
        kwargs,
        (config["beta1"], config["beta2"]),
        initialize=checkpoint is None,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=config["dtype"] == "float16")
    step = 0
    best_val_loss = float("inf")
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint["scaler"]:
            scaler.load_state_dict(checkpoint["scaler"])
        step = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]

    raw_model = model
    if config["compile"]:
        model = torch.compile(model)
    if ddp:
        model = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)
    torch.manual_seed(config["seed"] + rank)
    train_rng = torch.Generator().manual_seed(config["seed"] + rank)
    eval_rng = torch.Generator().manual_seed(config["seed"] + 10000 + rank)
    if checkpoint is not None and len(checkpoint["rng_states"]) == world_size:
        state = checkpoint["rng_states"][rank]
        torch.set_rng_state(state["torch"])
        train_rng.set_state(state["train"])
        eval_rng.set_state(state["eval"])
        if device.type == "cuda" and state["cuda"] is not None:
            torch.cuda.set_rng_state(state["cuda"], device)
    elif checkpoint is not None:
        logging.warning("World size changed; optimizer state resumes but rank RNG streams restart.")
    checkpoint = None
    wandb_run = None
    if config["wandb_log"] and rank == 0:
        import wandb

        wandb_run = wandb.init(project=config["wandb_project"], name=config["wandb_run_name"], config=config)

    def log_metrics(metrics: dict[str, Any]) -> None:
        if rank == 0:
            logging.info("%s", metrics)
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics) + "\n")
            if wandb_run is not None:
                wandb_run.log(metrics, step=step)

    @torch.no_grad()
    def evaluate() -> dict[str, float]:
        model.eval()
        losses = {}
        for split in ("train", "val"):
            total = torch.zeros((), device=device)
            for _ in range(config["eval_iters"]):
                x, y = data.batch(split, config["batch_size"], device, eval_rng)
                with autocast_context(device, config["dtype"]):
                    _, loss = model(x, y)
                total += loss.detach()
            if ddp:
                dist.all_reduce(total)
            losses[f"{split}/loss"] = total.item() / (config["eval_iters"] * world_size)
        model.train()
        return losses

    def save() -> None:
        state = {
            "torch": torch.get_rng_state(),
            "train": train_rng.get_state(),
            "eval": eval_rng.get_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        }
        states: list[Any] = [None] * world_size
        if ddp:
            dist.all_gather_object(states, state)
        else:
            states[0] = state
        if rank == 0:
            payload = {
                "model": raw_model.state_dict(),
                "model_args": model_args,
                "optimizer": optimizer.state_dict(),
                "emerging_optimizer": config["optimizer"],
                "scaler": scaler.state_dict(),
                "iter_num": step,
                "best_val_loss": best_val_loss,
                "config": config,
                "rng_states": states,
                "meta": data.meta,
            }
            temporary = output / "ckpt.pt.tmp"
            torch.save(payload, temporary)
            temporary.replace(output / "ckpt.pt")

    optimizer.zero_grad(set_to_none=True)
    if rank == 0:
        logging.info("Tokens per update: %d", config["batch_size"] * config["block_size"] * accumulation * world_size)
    # A resumed run continues directly; reevaluation would advance its saved evaluation RNG.
    if step == 0 or config["eval_only"]:
        losses = evaluate()
        best_val_loss = min(best_val_loss, losses["val/loss"])
        log_metrics({"iter": step, **losses})
    if not config["eval_only"]:
        while step < config["max_iters"]:
            if device.type == "cuda" and (step + 1) % config["log_interval"] == 0:
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            multiplier = lr_multiplier(step, config)
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * multiplier
            accumulated_loss = torch.zeros((), device=device)
            for micro_step in range(accumulation):
                sync = model.no_sync() if ddp and micro_step < accumulation - 1 else nullcontext()
                with sync:
                    x, y = data.batch("train", config["batch_size"], device, train_rng)
                    with autocast_context(device, config["dtype"]):
                        _, loss = model(x, y)
                    accumulated_loss += loss.detach() / accumulation
                    scaler.scale(loss / accumulation).backward()
            if config["grad_clip"]:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            # PSGD samples damping noise; DDP replicas must use identical optimizer randomness.
            rng_context = (
                optimizer_rng(device, config["seed"] + step + 1000000)
                if config["optimizer"] == "psgd_pro"
                else nullcontext()
            )
            with rng_context:
                scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % config["log_interval"] == 0:
                if ddp:
                    dist.all_reduce(accumulated_loss)
                loss_value = accumulated_loss.item() / world_size
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - start
                tokens = config["batch_size"] * config["block_size"] * accumulation * world_size
                log_metrics(
                    {
                        "iter": step,
                        "loss": loss_value,
                        "lr": config["learning_rate"] * multiplier,
                        "tokens_per_second": tokens / elapsed,
                        "step_ms": elapsed * 1000,
                    }
                )
            final = step == config["max_iters"]
            if step % config["eval_interval"] == 0 or final:
                losses = evaluate()
                improved = losses["val/loss"] < best_val_loss
                best_val_loss = min(best_val_loss, losses["val/loss"])
                log_metrics({"iter": step, **losses})
                if improved or config["always_save_checkpoint"] or final:
                    save()
        if config["max_iters"] == 0:
            save()
    if wandb_run is not None:
        wandb_run.finish()


def main(argv: list[str]) -> None:
    """Parse absl flags and launch training; --flagfile supports reusable recipes."""
    if len(argv) > 1:
        raise app.UsageError("Use named flags or --flagfile; Python configuration files are unsupported.")
    if FLAGS.list_optimizers:
        print("\n".join(sorted(optimizer_classes())))
        return
    train({key: FLAGS[key].value for key in _CONFIG_FLAGS})


if __name__ == "__main__":
    app.run(main)
