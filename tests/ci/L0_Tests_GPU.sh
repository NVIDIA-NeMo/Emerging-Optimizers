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
export TORCH_COMPILE_DISABLE=1
export CUDA_VISIBLE_DEVICES=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0

error=0
coverage run -p --source=emerging_optimizers tests/test_muon_utils.py -v -2 || error=1
coverage run -p --source=emerging_optimizers tests/test_adaptive_muon.py -v -2 || error=1
coverage run -p --source=emerging_optimizers tests/test_orthogonalized_optimizer.py -v -2 || error=1
coverage run -p --source=emerging_optimizers tests/test_spectron.py --device=cuda -v -2 || error=1
coverage run -p --source=emerging_optimizers tests/test_soap_utils.py -v -2 || error=1
coverage run -p --source=emerging_optimizers tests/test_soap.py -v -2 || error=1
coverage run -p --source=emerging_optimizers tests/soap_mnist_test.py -v -2 || error=1
coverage run -p --source=emerging_optimizers tests/test_scalar_optimizers.py --device=cuda -v -2 || error=1
coverage run -p --source=emerging_optimizers tests/test_spectral_clipping_utils.py -v -2  || error=1
coverage run -p --source=emerging_optimizers tests/test_triton_kernels.py TsyrkIntegerInputTest -v -2  || error=1
coverage run -p --source=emerging_optimizers tests/test_normalized_optimizer.py --device=cuda  -v -2 || error=1
coverage run -p --source=emerging_optimizers tests/normalized_optimizer_convergence_test.py --device=cuda  -v -2 || error=1
coverage run -p --source=emerging_optimizers tests/test_psgd_contractions.py --device=cuda  -v -2 || error=1
coverage run -p --source=emerging_optimizers tests/test_psgd_utils.py --device=cuda  -v -2 || error=1
coverage run -p --source=emerging_optimizers tests/test_psgd_convergence.py --device=cuda  -v -2 || error=1
coverage run -p --source=emerging_optimizers tests/test_utils_modules.py -v -2 || error=1
coverage run -p --source=emerging_optimizers tests/test_registry.py -v -2 || error=1

exit "${error}"
