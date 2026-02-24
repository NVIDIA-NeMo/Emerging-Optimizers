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
for test in `find tests -type f -name 'test_*' ! -name '*_cpu.py'`; do
    coverage run -p --source=emerging_optimizers $test --device=cuda -v -2 || error=1
done

coverage run -p --source=emerging_optimizers tests/convergence/soap_mnist_test.py -v -2 || error=1
coverage run -p --source=emerging_optimizers tests/convergence/normalized_optimizer_test.py --device=cuda  -v -2 || error=1

exit "${error}"
