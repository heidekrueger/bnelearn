import warnings
import pytest
import torch
from bnelearn.mechanism import PrisonersDilemma

"""Setup shared objects"""

gpu_device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test_2_player_matrix_gpu():
    """
    This test only applies to a setting where cuda is available!
    Tests
    - whether 2 player matrix game works expected
      * should accept both cpu and gpu action tensors and return results on gpu
      * should return valid results
      * Should fail graciously when given invalid input.
    """
    if gpu_device == 'cpu':
        pytest.skip(msg='Cuda not available. skipping.')

    game = PrisonersDilemma(cuda = True)
    actions_pure_cpu = torch.tensor([
        [[0], [0]],
        [[0], [1]],
        [[1], [0]],
        [[1], [1]]
    ])

    actions_pure_gpu = actions_pure_cpu.to(gpu_device)

    allocation, payments = game.play(actions_pure_gpu)

    # device
    assert allocation.device.type == 'cuda', "Result should be on GPU!"
    assert payments.device.type == 'cuda', "Result should be on GPU!"

    # return shapes
    assert allocation.shape == torch.Size([4, 2, 1]), \
        "Invalid allocation shape! Should be batch x n_players x items"
    assert payments.shape == torch.Size([4, 2]), \
        "Invalid payments shape. Should be batch x n_players"

    # allocations should be zero
    assert torch.equal(allocation, torch.zeros_like(allocation)), \
        "All allocations in matrix game should be 0"
    assert torch.equal(payments, -game.outcomes.view(4,2)), \
        "incorrect payments. Should be negative of outcome."
    
    actions_mixed = torch.tensor([
        [[.5], [.5]],
        [[0.], [1.]],
        [[.33], [.67]],
        [[1], [1.]]
    ])

    with pytest.raises(AssertionError):
        game.play(actions_mixed)
        pytest.fail("Game class should validate input!")

    actions_invalid_shape = torch.tensor([[1], [5], [3]]) 

    with pytest.raises(AssertionError):
        game.play(actions_invalid_shape)
        pytest.fail("Game class should validate input!")
    


