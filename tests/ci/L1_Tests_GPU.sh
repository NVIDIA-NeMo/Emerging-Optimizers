# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
export CUDA_VISIBLE_DEVICES=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0

error=0
python tests/test_muon_utils.py || error=1
python tests/test_adaptive_muon.py || error=1
python tests/test_orthogonalized_optimizer.py || error=1
python tests/test_soap_utils.py || error=1
python tests/test_soap.py || error=1
python tests/test_scalar_optimizers.py --device=cuda || error=1
python tests/test_spectral_clipping_utils.py || error=1
python tests/test_triton_kernels.py || error=1
python tests/test_normalized_optimizer.py
python tests/normalized_optimizer_convergence_test.py || error=1

exit "${error}"
