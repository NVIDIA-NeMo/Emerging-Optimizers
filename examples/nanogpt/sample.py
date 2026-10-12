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
    meta = {}
    if FLAGS.data_dir:
        with (Path(FLAGS.data_dir) / "meta.pkl").open("rb") as stream:
            meta = pickle.load(stream)
    if FLAGS.init_from != "resume" and ("stoi" in meta or "itos" in meta):
        raise app.UsageError("Character metadata cannot be used with pretrained GPT-2; omit --data_dir.")
    upstream = load_nanogpt(FLAGS.nanogpt_dir)
    model: Any
    if FLAGS.init_from == "resume":
        checkpoint = load_checkpoint(Path(FLAGS.checkpoint))
        model = upstream.GPT(upstream.GPTConfig(**checkpoint["model_args"]))
        load_model_state(model, checkpoint["model"])
        saved_meta = checkpoint.get("meta", {})
        if not FLAGS.data_dir:
            meta = saved_meta
        elif "stoi" in saved_meta and meta.get("stoi") != saved_meta["stoi"]:
            raise app.UsageError("Character metadata does not match the checkpoint's saved mapping.")
    else:
        model = upstream.GPT.from_pretrained(FLAGS.init_from, {"dropout": 0.0})
    if "stoi" in meta or "itos" in meta:
        stoi, itos = meta.get("stoi"), meta.get("itos")
        if not isinstance(stoi, dict) or not isinstance(itos, dict):
            raise app.UsageError("Character metadata must include both stoi and itos dictionaries.")
        vocab_size = model.config.vocab_size
        if meta.get("vocab_size") != vocab_size or len(stoi) != vocab_size or set(itos) != set(range(vocab_size)):
            raise app.UsageError("Character metadata must cover exactly the loaded model's vocabulary.")
        if any(
            not isinstance(char, str) or len(char) != 1 or not isinstance(token, int) or itos.get(token) != char
            for char, token in stoi.items()
        ):
            raise app.UsageError("Character metadata stoi and itos must be inverse character mappings.")
        encode = lambda text: [stoi[char] for char in text]
        decode = lambda tokens: "".join(itos[token] for token in tokens)
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
