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


mkdir -p test-results

error=0
for test in `find tests -type f -name 'test_*' ! -name '*_cpu.py'`; do
    echo "Running $test with random seed"
    report_name="test-results/${test}.xml"
    coverage run -p --source=emerging_optimizers $test --device=cuda -v -2 --xml_output_file="$report_name" || error=1
done

fix_seed=42
for test in `find tests -type f -name 'test_*' ! -name '*_cpu.py'`; do
    echo "Running $test with fixed seed $fix_seed"
    report_name="test-results/${test}_seed${fix_seed}.xml"
    coverage run -p --source=emerging_optimizers $test --device=cuda --seed=$fix_seed -v -2 --xml_output_file="$report_name" || error=1
done

for test in `find tests/convergence -type f -name '*_test.py'`; do
    echo "Running $test with fixed seed"
    report_name="test-results/${test}_seed${fix_seed}.xml"
    coverage run -p --source=emerging_optimizers $test --device=cuda --seed=$fix_seed -v -2 || error=1
done


exit "${error}"
