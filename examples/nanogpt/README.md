# nanoGPT with Emerging Optimizers

Run from the repository root in a PyTorch container with `emerging-optimizers`
installed, a local [nanoGPT](https://github.com/karpathy/nanoGPT) checkout, and
prepared `train.bin`/`val.bin` data (`prepare.py` prepares character data).

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

Features:

- Imports the upstream GPT model directly; standalone scripts with absl flags and flagfiles.
- 27 optimizer choices, including Muown and AdamW; excludes TP REKLS and StackedSoap. See `--list_optimizers`.
- Matrix optimizers paired with AdamW for embeddings, output head, biases, and normalization parameters.
- Checkpoint resume, validation, and text generation through `sample.py`.
- Gradient accumulation, clipping, and cosine learning-rate decay with warmup.
- CPU/GPU training, FP32/BF16/FP16, optional `torch.compile`, and PyTorch DDP.
- Upstream binary datasets and character-data preparation without NumPy.
- Console and JSONL metrics; optional W&B logging.
- No additional required dependencies beyond PyTorch and absl-py. Optional GPT-2 initialization, GPT-2 sampling, and W&B use `transformers`, `tiktoken`, and `wandb`, respectively.
