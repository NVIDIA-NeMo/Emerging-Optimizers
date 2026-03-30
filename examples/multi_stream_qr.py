"""Run multiple QR decompositions concurrently on different CUDA streams via cuSOLVER.

Each torch.linalg.qr call dispatches to cuSOLVER's geqrf + orgqr under the hood.
By placing each call on its own stream we allow the GPU to overlap the work when the
matrices are small enough that a single decomposition doesn't saturate the device.
"""

import time
from typing import Any

import torch
from absl import app, flags


FLAGS = flags.FLAGS

flags.DEFINE_integer("m", 1024, "Number of rows per matrix.")
flags.DEFINE_integer("n", 512, "Number of columns per matrix.")
flags.DEFINE_integer("num_matrices", 8, "Number of matrices to decompose.")
flags.DEFINE_integer("num_streams", 8, "Number of CUDA streams to use.")
flags.DEFINE_bool("use_cuda_graph", False, "Capture and replay with CUDA graphs.")
flags.DEFINE_integer("warmup_iters", 3, "Warmup iterations before timing.")


def multi_stream_qr(
    matrices: list[torch.Tensor],
    streams: list[torch.cuda.Stream],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Compute QR decomposition of each matrix on a separate CUDA stream.

    Args:
        matrices: List of 2-D CUDA tensors.
        streams: List of CUDA streams to round-robin across.

    Returns:
        List of (Q, R) tuples, one per input matrix.
    """
    if not matrices:
        return []

    num = len(matrices)

    # Placeholder results
    results: list[tuple[torch.Tensor, torch.Tensor]] = [(torch.empty(0), torch.empty(0)) for _ in range(num)]

    # Launch each QR on its assigned stream
    capturing_stream = torch.cuda.current_stream()
    for i, mat in enumerate(matrices):
        stream = streams[i % len(streams)]
        stream.wait_stream(capturing_stream)
        with torch.cuda.stream(stream):
            q, r = torch.linalg.qr(mat)
            results[i] = (q, r)

    # Join child streams back to the current stream
    for stream in streams:
        capturing_stream.wait_stream(stream)

    return results


def main(_: Any) -> None:
    """Run the benchmark."""
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")

    m = FLAGS.m
    n = FLAGS.n
    num_matrices = FLAGS.num_matrices
    num_streams = FLAGS.num_streams

    matrices = [torch.randint(1, 10, (m, n), device=device, dtype=torch.float32) for _ in range(num_matrices)]

    use_cuda_graph = FLAGS.use_cuda_graph
    warmup_iters = FLAGS.warmup_iters

    # Warmup
    for _ in range(warmup_iters):
        _ = torch.linalg.qr(matrices[0])
    torch.cuda.synchronize()

    # --- Single-stream benchmark ---
    def run_single() -> list[tuple[torch.Tensor, torch.Tensor]]:
        results = []
        for mat in matrices:
            results.append(torch.linalg.qr(mat))
        return results

    if use_cuda_graph:
        # Warmup run for graph capture (CUDA graphs require a prior execution)
        run_single()
        torch.cuda.synchronize()

        single_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(single_graph):
            single_results = run_single()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        single_graph.replay()
        torch.cuda.synchronize()
        single_ms = (time.perf_counter() - t0) * 1000
    else:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        single_results = run_single()
        torch.cuda.synchronize()
        single_ms = (time.perf_counter() - t0) * 1000

    # --- Multi-stream benchmark ---
    streams = [torch.cuda.Stream() for _ in range(num_streams)]

    def run_multi() -> list[tuple[torch.Tensor, torch.Tensor]]:
        return multi_stream_qr(matrices, streams)

    if use_cuda_graph:
        run_multi()
        torch.cuda.synchronize()

        multi_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(multi_graph):
            multi_results = run_multi()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        multi_graph.replay()
        torch.cuda.synchronize()
        multi_ms = (time.perf_counter() - t0) * 1000
    else:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        multi_results = run_multi()
        torch.cuda.synchronize()
        multi_ms = (time.perf_counter() - t0) * 1000

    max_err = max(((mq @ mr - matrices[i]) / matrices[i]).abs().max() for i, (mq, mr) in enumerate(multi_results))
    print(f"Max reconstruction error (multi-stream): {max_err:.5e}")
    max_err = max(((sq @ sr - matrices[i]) / matrices[i]).abs().max() for i, (sq, sr) in enumerate(single_results))
    print(f"Max reconstruction error (single-stream): {max_err:.5e}")

    print(f"Matrices : {num_matrices} x ({m}, {n})")
    print(f"Streams  : {num_streams}")
    print(f"Single-stream : {single_ms:.2f} ms")
    print(f"Multi-stream  : {multi_ms:.2f} ms")
    print(f"Speedup       : {single_ms / multi_ms:.2f}x")
    print(f"CUDA graph    : {'yes' if use_cuda_graph else 'no'}")


if __name__ == "__main__":
    app.run(main)
