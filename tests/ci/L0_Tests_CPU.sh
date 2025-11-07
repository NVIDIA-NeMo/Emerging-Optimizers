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

error=0
torchrun --nproc_per_node=8 --no-python coverage run -p tests/test_distributed_muon_utils_cpu.py  -v -2  || error=1
torchrun --nproc_per_node=4 --no-python coverage run -p tests/test_distributed_muon_utils_cpu.py  -v -2 || error=1
coverage run -p --source=emerging_optimizers tests/test_scalar_optimizers.py --device=cpu  -v -2 || error=1
coverage run -p --source=emerging_optimizers tests/test_procrustes_step.py --device=cpu  -v -2 || error=1

exit "${error}"
