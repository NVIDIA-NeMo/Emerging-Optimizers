# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate text with upstream GPT.generate and absl flags."""

import pickle
from pathlib import Path
from typing import Any

import torch
from absl import app, flags
from common import autocast_context, load_checkpoint, load_model_state, load_nanogpt


FLAGS = flags.FLAGS
flags.DEFINE_string("nanogpt_dir", "../nanoGPT", "Upstream nanoGPT checkout.")
flags.DEFINE_string("checkpoint", "out/nanogpt/ckpt.pt", "Checkpoint for init_from=resume.")
flags.DEFINE_enum("init_from", "resume", ["resume", "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"], "Initialization.")
flags.DEFINE_string("data_dir", None, "Optional upstream dataset directory containing character meta.pkl.")
flags.DEFINE_string("start", "\n", "Prompt, or FILE:path to read a prompt file.")
flags.DEFINE_integer("num_samples", 1, "Number of samples.", lower_bound=1)
flags.DEFINE_integer("max_new_tokens", 200, "Maximum new tokens per sample.", lower_bound=0)
flags.DEFINE_float("temperature", 0.8, "Sampling temperature; must be positive.")
flags.DEFINE_integer("top_k", 200, "Top-k filtering; zero disables it.", lower_bound=0)
flags.DEFINE_integer("seed", 1337, "Sampling seed.")
flags.DEFINE_string("device", "cuda", "cpu, cuda, or cuda:N.")
flags.DEFINE_enum("dtype", "bfloat16", ["float32", "bfloat16", "float16"], "Inference precision.")
flags.DEFINE_bool("compile", False, "Compile the model.")


def main(argv: list[str]) -> None:
    """Load a trained or pretrained model and reuse upstream generation."""
    if len(argv) > 1:
        raise app.UsageError("Use named flags; positional arguments are unsupported.")
    if FLAGS.temperature <= 0:
        raise app.UsageError("temperature must be positive")
    torch.manual_seed(FLAGS.seed)
    device = torch.device(FLAGS.device)
    upstream = load_nanogpt(FLAGS.nanogpt_dir)
    meta = {}
    model: Any
    if FLAGS.init_from == "resume":
        checkpoint = load_checkpoint(Path(FLAGS.checkpoint))
        model = upstream.GPT(upstream.GPTConfig(**checkpoint["model_args"]))
        load_model_state(model, checkpoint["model"])
        meta = checkpoint.get("meta", {})
    else:
        model = upstream.GPT.from_pretrained(FLAGS.init_from, {"dropout": 0.0})
    if FLAGS.data_dir:
        with (Path(FLAGS.data_dir) / "meta.pkl").open("rb") as stream:
            meta = pickle.load(stream)
    if "stoi" in meta:
        encode = lambda text: [meta["stoi"][char] for char in text]
        decode = lambda tokens: "".join(meta["itos"][token] for token in tokens)
    else:
        import tiktoken

        encoding = tiktoken.get_encoding("gpt2")
        encode = lambda text: encoding.encode(text, allowed_special={"<|endoftext|>"})
        decode = encoding.decode
    prompt = Path(FLAGS.start[5:]).read_text() if FLAGS.start.startswith("FILE:") else FLAGS.start
    tokens = encode(prompt)
    if not tokens:
        raise app.UsageError("The prompt must contain at least one token.")
    x = torch.tensor([tokens], device=device, dtype=torch.long)
    model.eval().to(device)
    if FLAGS.compile:
        model = torch.compile(model)
    with torch.no_grad(), autocast_context(device, FLAGS.dtype):
        for _ in range(FLAGS.num_samples):
            result = model.generate(x, FLAGS.max_new_tokens, temperature=FLAGS.temperature, top_k=FLAGS.top_k or None)
            print(decode(result[0].tolist()))


if __name__ == "__main__":
    app.run(main)
