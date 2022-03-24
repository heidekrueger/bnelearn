""" This test file checks whether NeuralNetStrategy initializations have the correct behavior."""

import time

import pytest
import torch

from bnelearn.strategy import NeuralNetStrategy

batch_size = 8

# each test input takes form input_length:int, output_length: int, hidden_nodes: list[int]
ids, testdata = zip(*[
    ['1551 - standard', (1,1, [5,5])],
    ['2552 - multi-dimensional inputs and outputs', (2,2, [5,5])],
    ['2551 - different output than input', (2,1,[5,5])],
    ['21 - no hidden layers', (2,1,[])]
])


def assert_nn_initialization(input_length, output_length,
                             hidden_nodes, device):
    """Initializes a specific nn on a specific device."""

    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip("CUDA not available. skipping...")
    
    hidden_activations = [torch.nn.SELU() for _ in hidden_nodes]

    s = NeuralNetStrategy(
        input_length = input_length,
        output_length = output_length,
        hidden_nodes = hidden_nodes,
        hidden_activations = hidden_activations
    ).to(device)

    input_tensor = torch.ones(batch_size, input_length, device = device)

    assert s(input_tensor).shape == torch.Size([batch_size, output_length]), \
        "NN initialization failed!"

@pytest.mark.parametrize("input_length,output_length,hidden_nodes", testdata, ids=ids)
def test_nn_initialization(input_length, output_length, hidden_nodes):
    """Tests whether nn init works."""

    assert_nn_initialization(input_length, output_length, hidden_nodes, 'cpu')
    assert_nn_initialization(input_length, output_length, hidden_nodes, 'cuda')

# TODO: tests for pretraining