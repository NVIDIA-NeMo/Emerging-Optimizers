# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from unittest import mock

import torch
from absl import flags, logging
from absl.testing import absltest

from emerging_optimizers.utils import get_pg_rank, get_pg_size


flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to run tests on")
flags.DEFINE_integer("seed", None, "Random seed for reproducible tests")
FLAGS = flags.FLAGS


def setUpModule() -> None:
    if FLAGS.seed is not None:
        logging.info("Setting random seed to %d", FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FLAGS.seed)


class GetPgSizeRankTest(absltest.TestCase):
    def setUp(self):
        self.assertFalse(torch.distributed.is_initialized())
        self.group = mock.create_autospec(torch.distributed.ProcessGroup, instance=True)
        self.group.size.return_value = 4
        self.group.rank.return_value = 3

    def test_none_group_uninitialized_returns_defaults(self):
        self.assertEqual(get_pg_size(), 1)
        self.assertEqual(get_pg_rank(), 0)
        self.assertEqual(get_pg_size(None), 1)
        self.assertEqual(get_pg_rank(None), 0)

    def test_group_uninitialized_returns_defaults(self):
        self.assertEqual(get_pg_size(self.group), 1)
        self.assertEqual(get_pg_rank(self.group), 0)
        self.group.size.assert_not_called()
        self.group.rank.assert_not_called()

    def test_none_group_initialized_returns_defaults(self):
        with mock.patch("torch.distributed.is_initialized", return_value=True):
            self.assertEqual(get_pg_size(None), 1)
            self.assertEqual(get_pg_rank(None), 0)

    def test_group_initialized_queries_group(self):
        with mock.patch("torch.distributed.is_initialized", return_value=True):
            self.assertEqual(get_pg_size(self.group), 4)
            self.assertEqual(get_pg_rank(self.group), 3)
        self.group.size.assert_called_once_with()
        self.group.rank.assert_called_once_with()


if __name__ == "__main__":
    absltest.main()
