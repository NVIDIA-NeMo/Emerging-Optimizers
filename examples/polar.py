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
import numpy as np
import nvmath
import torch
from nvmath.bindings import cusolverDn
from scipy import linalg


# cublasFillMode_t constants
CUBLAS_FILL_MODE_LOWER = 0
CUBLAS_FILL_MODE_UPPER = 1
CUBLAS_FILL_MODE_FULL = 2


def polar_via_cusolver(a_torch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Polar Decomposition A = U * H using cusolverDnXpolar.

    Args:
        a_torch: Input tensor of shape (m, n) on CUDA, requires m >= n.

    Returns:
        U: Unitary matrix of shape (m, n).
        H: Hermitian positive semi-definite matrix of shape (n, n).
    """
    m, n = a_torch.shape
    device = a_torch.device

    # Map torch dtype to CUDA data type.
    dtype_map = {
        torch.float32: nvmath.CudaDataType.CUDA_R_32F,
        torch.float64: nvmath.CudaDataType.CUDA_R_64F,
    }
    cuda_dtype = dtype_map[a_torch.dtype]

    # cuSOLVER expects column-major layout. Create column-major internal buffers.
    A = torch.empty(n, m, dtype=a_torch.dtype, device=device).t()
    A.copy_(a_torch)
    H = torch.empty(n, n, dtype=a_torch.dtype, device=device).t()
    H.zero_()

    lda = m
    ldh = n

    # Setup handle, params, and stream.
    handle = cusolverDn.create()
    params = cusolverDn.create_params()
    stream = torch.cuda.current_stream().cuda_stream
    cusolverDn.set_stream(handle, stream)

    # Query workspace sizes (returns device bytes and host bytes).
    d_work_size, h_work_size = cusolverDn.xpolar_buffer_size(
        handle,
        params,
        CUBLAS_FILL_MODE_FULL,
        n,
        m,
        cuda_dtype,
        A.data_ptr(),
        lda,
        cuda_dtype,
        H.data_ptr(),
        ldh,
        cuda_dtype,
    )

    # Allocate device and host workspaces.
    d_work = torch.empty(d_work_size, dtype=torch.uint8, device=device)
    h_work = np.empty(h_work_size, dtype=np.uint8)

    # Diagnostic outputs (on device, float64).
    d_res_nrm = torch.empty(1, dtype=torch.float64, device=device)
    d_A_nrmF = torch.empty(1, dtype=torch.float64, device=device)
    d_rcond = torch.empty(1, dtype=torch.float64, device=device)
    d_info = torch.zeros(1, dtype=torch.int32, device=device)

    # Execute polar decomposition. A is overwritten with U in-place.
    cusolverDn.xpolar(
        handle,
        params,
        CUBLAS_FILL_MODE_FULL,
        m,
        n,
        cuda_dtype,
        A.data_ptr(),
        lda,
        cuda_dtype,
        H.data_ptr(),
        ldh,
        cuda_dtype,
        d_work.data_ptr(),
        d_work_size,
        h_work.ctypes.data,
        h_work_size,
        d_res_nrm.data_ptr(),
        d_A_nrmF.data_ptr(),
        d_rcond.data_ptr(),
        d_info.data_ptr(),
    )

    torch.cuda.synchronize()

    if d_info.item() != 0:
        raise RuntimeError(f"cusolverDnXpolar failed with info={d_info.item()}")

    cusolverDn.destroy_params(params)
    cusolverDn.destroy(handle)

    # Convert column-major results back to row-major (contiguous).
    return A.contiguous(), H.contiguous()


if __name__ == "__main__":
    m, n = 128, 128
    A = torch.randn(m, n, device="cuda")

    U, H = polar_via_cusolver(A)

    U_ref, H_ref = linalg.polar(A.cpu().numpy())
    U_ref = torch.from_numpy(U_ref).to(A.device)
    H_ref = torch.from_numpy(H_ref).to(A.device)

    print("=== cuSOLVER ===")
    print(f"Reconstruction Error (A - U@H): {torch.norm(A - U @ H).item():.4g}")
    print(f"Orthogonality Error (U^T@U - I): {torch.norm(U.T @ U - torch.eye(n, device='cuda')).item():.4g}")

    print("\n=== scipy ===")
    print(f"Reconstruction Error (A - U@H): {torch.norm(A - U_ref @ H_ref).item():.4g}")
    print(f"Orthogonality Error (U^T@U - I): {torch.norm(U_ref.T @ U_ref - torch.eye(n, device='cuda')).item():.4g}")

    print("\n=== Difference ===")
    print(f"U diff: {torch.norm(U - U_ref).item():.4g}")
    print(f"H diff: {torch.norm(H - H_ref).item():.4g}")
