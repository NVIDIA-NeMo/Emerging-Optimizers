# nanoGPT with Emerging Optimizers

This example imports `GPT` and `GPTConfig` directly from a local
[karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) checkout. It does not copy
or modify the upstream model. The local training loop uses **absl flags** and
supports optimizer selection, scratch/pretrained initialization, checkpoint
resume, evaluation, gradient accumulation, clipping, cosine decay with warmup,
FP32/BF16/FP16, optional `torch.compile`, and `torchrun` DDP. Sampling calls
upstream `GPT.generate`.

The default character-training path requires only this repository's existing
runtime dependencies: **PyTorch and absl-py**, on Python 3.12+. Data loading uses
PyTorch memory mapping, and preparation uses the standard library. NumPy,
Transformers, datasets, tiktoken, and W&B are not required for this path.

## Run in a container

Use an existing PyTorch container with the repository and an upstream checkout
mounted. No virtual environment is needed. Run commands from this repository's
root inside the container, using its installed Python and PyTorch. The examples
are standalone scripts with sibling imports, not a Python package. To use the
repository source without installing it, add the root to the import path:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

This is unnecessary if `emerging-optimizers` is already installed in the container.

Obtain upstream once, outside or inside the container:

```bash
git clone https://github.com/karpathy/nanoGPT.git ../nanoGPT
git -C ../nanoGPT checkout 3adf61e154c3fe3fca428ad6bc3818b27a3b8291
```

That is the upstream revision used for validation. `--nanogpt_dir` names the
directory containing its `model.py`; the example never downloads executable
model code at runtime. Importing upstream `train.py` would immediately start its
training loop and execute its configurator, so only `model.py` is imported.

For example, start a container using an available PyTorch image:

```bash
docker run --rm -it --gpus all --ipc=host -e PYTHONPATH=/workspace \
  -v "$PWD":/workspace -v "$(realpath ../nanoGPT)":/opt/nanoGPT:ro \
  -w /workspace <pytorch-image> bash
```

Prepare local text, or omit `--input_file` to download Tiny Shakespeare:

```bash
python examples/nanogpt/prepare.py \
  --input_file=/path/to/input.txt --data_dir=/tmp/shakespeare_char
```

Train a small model; substitute any name from `--list_optimizers`:

```bash
python examples/nanogpt/train.py \
  --nanogpt_dir=/opt/nanoGPT --data_dir=/tmp/shakespeare_char \
  --out_dir=/tmp/nanogpt-muon --optimizer=muon \
  --n_layer=4 --n_head=4 --n_embd=128 --block_size=128 \
  --batch_size=8 --gradient_accumulation_steps=4 \
  --learning_rate=0.003 --adamw_learning_rate=0.0003 \
  --optimizer_kwargs='{"momentum":0.95,"extra_scale_factor":0.2}' \
  --max_iters=2000 --warmup_iters=100 --lr_decay_iters=2000 \
  --eval_interval=100 --dtype=bfloat16
```

These settings demonstrate usage; they are not convergence-tuned presets. Use
`--device=cpu --dtype=float32` for a CPU smoke run with a much smaller model.
Use `--flagfile=recipe.flags` to keep one `--flag=value` per line. Upstream Python
config files are replaced by absl flagfiles.

## Optimizer coverage

```bash
python examples/nanogpt/train.py --list_optimizers
```

Supported names:

- Scalar: `adamw`, `lion`, `signum`, `laprop`, `sim_ademamix`.
- Matrix: `muon`, `adaptive_muon`, `muon_hyperball`, `muown`, `mop`, `scion`,
  `polargrad`, `spel`, `soap`, `moso`, `rekls`, `shampoo`, `kl_shampoo`, `kl_soap`,
  `kl_m_soap`, `rekls_v3`, `psgd_pro`, `iso`, `oblique_sgd`, `oblique_adam`,
  `oblique_steepest_sgd`, `oblique_steepest_adam`.

Tensor-parallel REKLS and StackedSoap are excluded. This covers 25 registered
optimizers, contributed Muown, and the PyTorch AdamW baseline.

Scalar optimizers own all parameters. Matrix optimizers own the hidden 2D
weights; auxiliary AdamW owns token/position embeddings, the output head,
biases, and normalization parameters. Tied embeddings are assigned exactly
once. Biases and normalization scales have zero weight decay. Matrix and
auxiliary learning rates retain their ratio throughout the schedule.

`--optimizer_kwargs` passes validated JSON options to the selected optimizer.
Its defaults are otherwise preserved, except that AdaptiveMuon gets momentum
0.95 when unspecified. `--beta1` and `--beta2` configure AdamW; pass `betas` in
JSON for another optimizer. Iso has no weight-decay option, so
`--weight_decay` only affects its auxiliary AdamW. Scion retains its own
Frank-Wolfe update semantics.

MuonHyperball rescales hidden matrices **before a fresh run** to the requested
`hyperball_radius` (default 1.0). Resuming preserves the saved weights. Oblique,
SPEL, and Iso also have different geometric constraints; selecting them is not
a guarantee that GPT's default initialization or a shared learning rate is
optimal. Fused QKV is optimized as one matrix, as stored by upstream nanoGPT.
SOAP/Shampoo-style optimizer state can be large; start with a small model.

## Resume, evaluation, and generation

`ckpt.pt` contains the model, both optimizer states where applicable, scaler,
completed iteration count, configuration, character metadata, and per-rank RNG
states. A final checkpoint is always written after training. Periodic saves
follow `--always_save_checkpoint` (otherwise only validation improvements save).
Metrics are appended to `metrics.jsonl`.

Resume using the same dataset and context length:

```bash
python examples/nanogpt/train.py \
  --nanogpt_dir=/opt/nanoGPT --data_dir=/tmp/shakespeare_char \
  --out_dir=/tmp/nanogpt-muon --init_from=resume \
  --block_size=128 --batch_size=8 --gradient_accumulation_steps=4 \
  --max_iters=3000
```

`max_iters` is the total number of updates, not additional updates. Resume
restores optimizer options and the saved LR schedule; changing `max_iters`
does not restart or stretch that schedule. Keep batch size, accumulation,
precision, world size, and evaluation settings unchanged for reproducibility.
Use `--checkpoint=/path/to/ckpt.pt` to resume from a different output directory.
Training resumes this example's checkpoints; original upstream checkpoints are
accepted for sampling. A resumed model's context length cannot be changed
because that would invalidate optimizer state.

Add `--eval_only` to evaluate once, including a resumed checkpoint at a nonzero
iteration. Generate character text without a tokenizer dependency:

```bash
python examples/nanogpt/sample.py \
  --nanogpt_dir=/opt/nanoGPT --checkpoint=/tmp/nanogpt-muon/ckpt.pt \
  --start='ROMEO:' --num_samples=2 --max_new_tokens=200 \
  --temperature=0.8 --top_k=50
```

Prompts must use the dataset's characters. `--start=FILE:prompt.txt` reads a
prompt file. For upstream character checkpoints, provide `--data_dir` so that
sampling can load their `meta.pkl`.

## Optional upstream features

- **DDP:** launch `torchrun --standalone --nproc_per_node=2
  examples/nanogpt/train.py ...`. `gradient_accumulation_steps` is global and must
  be divisible by world size, matching upstream. Evaluation averages across
  ranks. PSGD uses a shared optimizer RNG stream so its damping noise is identical
  across replicas while dropout and data RNG remain independent. CPU DDP uses `--device=cpu --backend=gloo --dtype=float32`.
- **Compilation:** `--compile` compiles the model. Some optimizer functions
  independently use `torch.compile`; set `TORCH_COMPILE_DISABLE=1` for fully
  eager debugging or quick smoke tests.
- **FP16:** `--dtype=float16` uses one GradScaler over the combined optimizer,
  so overflow skips both matrix and auxiliary updates together.
- **GPT-2 initialization:** `--init_from=gpt2` (or its size variants) calls
  upstream `GPT.from_pretrained` and requires optional `transformers`. Use a
  GPT-2-tokenized dataset, not the character dataset. Pretrained context length
  can be cropped before creating optimizer state.
- **GPT-2 sampling:** absent character metadata, sampling uses optional
  `tiktoken`, matching upstream GPT-2 tokenization.
- **W&B:** `--wandb_log --wandb_project=...` requires optional `wandb`. Console
  and JSONL logging work without it.
- **Existing datasets:** upstream little-endian uint16 `train.bin`/`val.bin`
  files and optional `meta.pkl` work directly. Default vocabulary size is
  50304 when metadata is absent; override with `--vocab_size` if needed.

There is no new training-framework dependency. DDP uses PyTorch directly;
tensor parallelism is outside this example's scope.

## Validation

Run against the mounted upstream checkout in the container:

```bash
TORCH_COMPILE_DISABLE=1 OMP_NUM_THREADS=1 python tests/test_nanogpt.py \
  --nanogpt_dir=/opt/nanoGPT --device=cpu
TORCH_COMPILE_DISABLE=1 OMP_NUM_THREADS=1 python tests/test_nanogpt.py \
  --nanogpt_dir=/opt/nanoGPT --device=cuda
```

The tests exercise actual upstream GPT forward/backward passes for every listed
optimizer, parameter ownership, finite updates, state restoration, unsigned data
loading, training/resume equivalence, evaluation, and generation. GPU optimizer
tests use BF16 autocast and also check FP16 overflow behavior. Model integration
tests skip if no checkout is supplied. These are compatibility tests, not
convergence or throughput benchmarks.
