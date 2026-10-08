# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare upstream-compatible character data using only the standard library."""

import array
import pickle
import sys
import urllib.request
from pathlib import Path

from absl import app, flags, logging


FLAGS = flags.FLAGS
flags.DEFINE_string("input_file", None, "Local UTF-8 input text; otherwise download Tiny Shakespeare.")
flags.DEFINE_string("data_dir", "data/shakespeare_char", "Output dataset directory.")
flags.DEFINE_float("train_fraction", 0.9, "Fraction of text used for training.", lower_bound=0.0, upper_bound=1.0)


def prepare(text: str, directory: Path, train_fraction: float = 0.9) -> None:
    """Encode text to little-endian uint16 bins and upstream's character metadata."""
    chars = sorted(set(text))
    if not chars or len(chars) > 65536:
        raise ValueError("Character vocabulary must contain between 1 and 65536 entries.")
    boundary = int(len(text) * train_fraction)
    if not 0 < boundary < len(text):
        raise ValueError("Training and validation splits must both be nonempty.")
    stoi = {char: index for index, char in enumerate(chars)}
    directory.mkdir(parents=True, exist_ok=True)
    for split, content in (("train", text[:boundary]), ("val", text[boundary:])):
        tokens = array.array("H", (stoi[char] for char in content))
        if sys.byteorder != "little":
            tokens.byteswap()
        (directory / f"{split}.bin").write_bytes(tokens.tobytes())
    with (directory / "meta.pkl").open("wb") as stream:
        pickle.dump({"vocab_size": len(chars), "stoi": stoi, "itos": dict(enumerate(chars))}, stream)
    logging.info(
        "Wrote %d training tokens, %d validation tokens, vocabulary %d", boundary, len(text) - boundary, len(chars)
    )


def main(argv: list[str]) -> None:
    """Prepare local text or the same Tiny Shakespeare corpus used by nanoGPT."""
    if len(argv) > 1:
        raise app.UsageError("Use named flags; positional arguments are unsupported.")
    if FLAGS.input_file:
        text = Path(FLAGS.input_file).read_text(encoding="utf-8")
    else:
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with urllib.request.urlopen(url, timeout=60) as response:
            text = response.read().decode("utf-8")
    prepare(text, Path(FLAGS.data_dir), FLAGS.train_fraction)


if __name__ == "__main__":
    app.run(main)
