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
CUDA_VISIBLE_DEVICES="0" coverage run -p --source=emerging_optimizers tests/test_muon_utils.py
CUDA_VISIBLE_DEVICES="0" coverage run -p --source=emerging_optimizers tests/test_orthogonalized_optimizer.py
CUDA_VISIBLE_DEVICES="0" coverage run -p --source=emerging_optimizers tests/test_soap_functions.py
CUDA_VISIBLE_DEVICES="0" coverage run -p --source=emerging_optimizers tests/test_soap_utils.py
CUDA_VISIBLE_DEVICES="0" coverage run -p --source=emerging_optimizers tests/soap_smoke_test.py
CUDA_VISIBLE_DEVICES="0" coverage run -p --source=emerging_optimizers tests/soap_mnist_test.py
CUDA_VISIBLE_DEVICES="0" coverage run -p --source=emerging_optimizers tests/test_scalar_optimizers.py





