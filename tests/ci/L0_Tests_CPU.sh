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

mkdir -p test-results/tests/

error=0
for n in 8 4; do
    for test in tests/test_distributed_*_cpu.py; do
        report_base=$(basename "$test" .py)
        torchrun --nproc_per_node=$n --no-python coverage run -p \
            "$test" \
            --xml_output_file="test-results/tests/${report_base}_n${n}.xml" \
            -v -2  || error=1
    done
done

# Single-process sanity check for TpRekls — exercises the size-1 tp_group path.
torchrun --nproc_per_node=1 --no-python coverage run -p \
    tests/test_distributed_rekls_cpu.py \
    --xml_output_file="test-results/tests/test_distributed_rekls_cpu_n1.xml" \
    -v -2  || error=1

for test in "tests/test_scalar_optimizers.py" "tests/test_procrustes_step.py"; do
    report_name="test-results/${test}.xml"
    coverage run -p --source=emerging_optimizers $test --device=cpu  -v -2 --xml_output_file="$report_name" || error=1
done

exit "${error}"
