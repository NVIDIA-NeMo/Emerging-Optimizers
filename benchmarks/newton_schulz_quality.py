# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate Newton-Schulz quality on synthetic Kimi-K3 routed-expert matrices.

Run from the repository root with ``python -m benchmarks.newton_schulz_quality``.
See docs/primer/newton-schulz-quality.md for the experiment and interpretation.
"""

import json
import math
from pathlib import Path
from typing import Any

import torch
from absl import app, flags

from emerging_optimizers.orthogonalized_optimizers.muon_utils import _COEFFICIENT_SETS, newton_schulz
from emerging_optimizers.utils import fp32_matmul_precision


# Logical, unpacked (out_features, in_features), not the MXFP4 storage shapes.
# https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json
# Transposing the same input preserves the quality metrics; evaluate one orientation.
EXPERT_SHAPES = {"gate_up": (3072, 3584)}
FLAGS = flags.FLAGS


def make_matrix(shape: tuple[int, int], spectrum: str, seed: int, device: str) -> torch.Tensor:
    """Create an FP32 Gaussian, uniform, or rectangular Hilbert matrix."""
    generator = torch.Generator(device=device).manual_seed(seed)
    if spectrum == "gaussian":
        return torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
    if spectrum == "uniform":
        return torch.rand(shape, generator=generator, device=device, dtype=torch.float32)
    if spectrum == "hilbert":
        rows = torch.arange(shape[0], device=device, dtype=torch.float32)[:, None]
        columns = torch.arange(shape[1], device=device, dtype=torch.float32)[None, :]
        return 1.0 / (rows + columns + 1.0)
    raise ValueError(f"Unknown input distribution: {spectrum}")


def polar_reference(x: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Compute the FP64 polar factor of the same FP32 input and its residual."""
    if not hasattr(torch.linalg, "polar"):
        raise RuntimeError("This experiment requires torch.linalg.polar (PyTorch 2.14 or newer).")
    # torch.linalg.polar requires rows >= columns. The wide polar factor follows
    # by transposing the tall input's factor back; H belongs to the tall problem.
    wide = x.shape[0] < x.shape[1]
    tall = (x.mT if wide else x).double()
    q, h = torch.linalg.polar(tall)
    residual = (torch.linalg.vector_norm(q @ h - tall) / torch.linalg.vector_norm(tall)).item()
    return (q.mT if wide else q), residual


def quality_metrics(x: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    """Measure polar error, semi-orthogonality, direction, and scale in FP64."""
    x = x.double()
    reference = reference.double()
    rank = min(x.shape)
    gram = x @ x.mT if x.shape[0] <= x.shape[1] else x.mT @ x
    identity = torch.eye(rank, device=x.device, dtype=x.dtype)
    norm = torch.linalg.vector_norm(x)
    reference_norm = torch.linalg.vector_norm(reference)
    return {
        "relative_polar_error": (torch.linalg.vector_norm(x - reference) / reference_norm).item(),
        "orthogonality_error": (torch.linalg.vector_norm(gram - identity) / math.sqrt(rank)).item(),
        "cosine_similarity": ((x * reference).sum() / (norm * reference_norm)).item(),
        "rms_ratio": (norm / reference_norm).item(),
    }


def plot_results(rows: list[dict[str, Any]], path: Path) -> None:
    """Plot each matrix and spectrum separately, with precision encoded by line style."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = list(dict.fromkeys((row["weight"], row["spectrum"]) for row in rows))
    fig, axes = plt.subplots(len(cases), 2, figsize=(13, 3.4 * len(cases) + 1), squeeze=False)
    colors = dict(zip(_COEFFICIENT_SETS, plt.get_cmap("tab10").colors))
    styles = {"highest": "-", "high": ":", "medium": "--"}
    for axes_row, (weight, spectrum) in zip(axes, cases):
        selected = [row for row in rows if (row["weight"], row["spectrum"]) == (weight, spectrum)]
        for ax, metric in zip(axes_row, ("relative_polar_error", "orthogonality_error")):
            for coefficient in _COEFFICIENT_SETS:
                if coefficient in {"simple", "aol", "cans"}:
                    continue
                for precision in dict.fromkeys(row["precision"] for row in selected):
                    group = [
                        row for row in selected if row["coefficient"] == coefficient and row["precision"] == precision
                    ]
                    if not group:
                        continue
                    ax.semilogy(
                        [row["steps"] for row in group],
                        [row[metric] for row in group],
                        color=colors[coefficient],
                        linestyle=styles[precision],
                        marker=".",
                        linewidth=2.2,
                        markersize=7,
                        label=f"{coefficient} / {precision}",
                    )
            ax.set(title=f"{weight} {selected[0]['shape']} / {spectrum}", xlabel="NS steps", ylabel=metric)
            ax.set_xticks(sorted({row["steps"] for row in selected}))
            ax.grid(False, which="both")
            ax.set_facecolor("white")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        fontsize=14,
        frameon=False,
        handlelength=3,
        markerscale=1.5,
        labelspacing=0.6,
        columnspacing=2,
    )
    fig.tight_layout(rect=(0, 0, 1, 1 - 1.85 / fig.get_figheight()))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor="white", transparent=False)
    plt.close(fig)


@torch.no_grad()
def main(_: list[str]) -> None:
    """Sweep repository coefficient schedules and save reproducible quality measurements."""
    if not hasattr(torch.linalg, "polar"):
        raise RuntimeError("This experiment requires torch.linalg.polar (PyTorch 2.14 or newer).")
    steps = sorted(set(int(value) for value in FLAGS.steps))
    if not steps or min(steps) < 1:
        raise ValueError("--steps must contain positive integers.")
    if FLAGS.shape_divisor < 1 or any(
        size % FLAGS.shape_divisor for shape in EXPERT_SHAPES.values() for size in shape
    ):
        raise ValueError("--shape_divisor must be positive and divide both expert dimensions.")
    if not FLAGS.coefficients:
        raise ValueError("--coefficients must contain at least one coefficient set.")
    for coefficient in FLAGS.coefficients:
        if coefficient not in _COEFFICIENT_SETS:
            raise ValueError(f"Unknown coefficient set: {coefficient}")
    if "cubic5" in FLAGS.coefficients and 5 not in steps:
        raise ValueError("Include 5 in --steps when requesting the fixed cubic5 schedule.")
    if not FLAGS.precisions or any(precision not in ("highest", "high", "medium") for precision in FLAGS.precisions):
        raise ValueError("--precisions must contain highest, high, or medium.")
    if not FLAGS.spectra or any(spectrum not in ("gaussian", "uniform", "hilbert") for spectrum in FLAGS.spectra):
        raise ValueError("--spectra must contain gaussian, uniform, or hilbert.")

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    metadata = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": FLAGS.device,
        "device_name": torch.cuda.get_device_name() if FLAGS.device == "cuda" else "cpu",
        "seed": FLAGS.seed,
        "shape_divisor": FLAGS.shape_divisor,
        "input_dtype": "float32",
        "reference": "torch.linalg.polar",
        "reference_dtype": "float64",
        "metric_dtype": "float64",
        "steps": steps,
        "precisions": FLAGS.precisions,
        "input_distributions": FLAGS.spectra,
        "uniform_interval": "[0, 1)",
        "hilbert_definition": "H[i, j] = 1 / (i + j + 1), zero-based indices, constructed in float32",
        "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "coefficient_sets": {key: _COEFFICIENT_SETS[key] for key in FLAGS.coefficients},
    }
    print(json.dumps(metadata, indent=2), flush=True)
    rows = []
    references = []
    for weight, full_shape in EXPERT_SHAPES.items():
        shape = tuple(size // FLAGS.shape_divisor for size in full_shape)
        for spectrum in FLAGS.spectra:
            print(f"Building {weight} {shape} / {spectrum}", flush=True)
            x = make_matrix(shape, spectrum, FLAGS.seed, FLAGS.device)
            reference, residual = polar_reference(x)
            baseline = quality_metrics(reference, reference)
            if (
                not math.isfinite(residual)
                or not all(math.isfinite(value) for value in baseline.values())
                or residual > 1e-8
                or baseline["orthogonality_error"] > 1e-8
            ):
                raise RuntimeError(f"Polar reference failed validation: residual={residual}, metrics={baseline}")
            references.append(
                {
                    "weight": weight,
                    "shape": shape,
                    "spectrum": spectrum,
                    "reconstruction_error": residual,
                    **baseline,
                }
            )
            for precision in FLAGS.precisions:
                for coefficient in FLAGS.coefficients:
                    for count in [5] if coefficient == "cubic5" else steps:
                        with fp32_matmul_precision(precision):
                            result = newton_schulz(x, steps=count, coefficient_type=coefficient)
                        metrics = quality_metrics(result, reference)
                        if not all(math.isfinite(value) for value in metrics.values()):
                            raise RuntimeError(
                                f"Nonfinite result: {weight}/{spectrum}/{precision}/{coefficient}/{count}"
                            )
                        row = {
                            "weight": weight,
                            "shape": shape,
                            "spectrum": spectrum,
                            "precision": precision,
                            "coefficient": coefficient,
                            "steps": count,
                            **metrics,
                        }
                        rows.append(row)
                        print(
                            f"{weight:7s} {spectrum:8s} {precision:7s} {coefficient:13s} {count:2d} "
                            f"polar={metrics['relative_polar_error']:.4e} "
                            f"orth={metrics['orthogonality_error']:.4e}",
                            flush=True,
                        )
    output = Path(FLAGS.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"metadata": metadata, "references": references, "results": rows}, indent=2) + "\n")
    if FLAGS.plot:
        plot_results(rows, Path(FLAGS.plot))


if __name__ == "__main__":
    flags.DEFINE_enum("device", "cuda", ["cpu", "cuda"], "Device for inputs, NS, and polar reference.")
    flags.DEFINE_integer("seed", 0, "Random seed; identical inputs are reused across the sweep.")
    flags.DEFINE_list("steps", ["5", "8", "10", "16"], "NS step counts.")
    flags.DEFINE_list("coefficients", list(_COEFFICIENT_SETS), "Repository coefficient sets to evaluate.")
    flags.DEFINE_list("precisions", ["highest", "medium"], "NS matmul precisions; medium explicitly uses BF16.")
    flags.DEFINE_list(
        "spectra", ["gaussian", "uniform", "hilbert"], "Inputs: N(0, 1), uniform [0, 1), or Hilbert H[i,j]=1/(i+j+1)."
    )
    flags.DEFINE_integer("shape_divisor", 1, "Divide dimensions for smoke tests; 1 uses actual expert dimensions.")
    flags.DEFINE_string("output", "/tmp/newton-schulz-quality.json", "Output JSON path.")
    flags.DEFINE_string("plot", None, "Optional figure path; requires matplotlib.")
    app.run(main)
