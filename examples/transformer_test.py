# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl.testing import parameterized
from sequence_layers import test_util
import tensorflow.compat.v2 as tf

from . import transformer


class TransformerTest(test_util.SequenceLayerTest, parameterized.TestCase):

  @parameterized.parameters(False, True)
  def test_transformer_encoder(self, training):
    batch_size, target_length, target_dimension = 2, 4, 6
    # The target sequence to learn.
    x = self.random_sequence(batch_size, target_length, target_dimension)
    l = transformer.TransformerEncoder(
        num_layers=2,
        dimension=8,
        num_heads=2,
        max_horizon=8,
        max_future_horizon=8,
    )

    self.verify_contract(l, x, training=training, pad_nan=False)
    # Use flex for Einsum.
    self.verify_tflite_step(l, x, use_flex=True)

  @parameterized.parameters(False, True)
  def test_transformer_decoder(self, training):
    batch_size, source_length, source_dim = 2, 3, 5
    target_length, target_dimension = 4, 6
    source_name = 'source'
    # The target sequence to learn.
    x = self.random_sequence(batch_size, target_length, target_dimension)
    l = transformer.TransformerDecoder(
        source_name, num_layers=2, dimension=8, num_heads=2, max_horizon=8
    )

    # The encoder sequence to attend to.
    source = self.random_sequence(batch_size, source_length, source_dim)
    constants = {source_name: source}

    self.verify_contract(
        l, x, training=training, pad_nan=False, constants=constants
    )
    # Use flex for Einsum.
    self.verify_tflite_step(l, x, constants=constants, use_flex=True)


if __name__ == '__main__':
  tf.test.main()
