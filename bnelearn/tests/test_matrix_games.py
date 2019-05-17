import warnings
import pytest
import torch
from bnelearn.mechanism import PrisonersDilemma, RockPaperScissors, JordanGame

#Setup shared objects

gpu_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# setup games
pd = PrisonersDilemma(cuda = True)
pd_cpu = PrisonersDilemma(cuda = False)


# setup actions
actions_pd_cpu = torch.tensor([
        [[0], [0]],
        [[0], [1]],
        [[1], [0]],
        [[1], [1]]
    ])

actions_pd_gpu = actions_pd_cpu.to(gpu_device)

pd_allocation, pd_payments = pd.play(actions_pd_cpu)



def test_output_on_gpu():
    """If the game has cuda=True, outputs should be on gpu regardless of input."""
    if gpu_device == 'cpu':
        pytest.skip(msg='Cuda not available. skipping.')

    assert pd_allocation.device.type == 'cuda', "Result should be on GPU for cpu inputs!"
    assert pd_payments.device.type == 'cuda', "Result should be on GPU for cpu inputs!"

    # run again with gpu inputs
    allocation, payments = pd.play(actions_pd_gpu)

    assert allocation.device.type == 'cuda', "Result should be on GPU for gpu inputs!"
    assert payments.device.type == 'cuda', "Result should be on GPU for gpu inputs!"

def test_output_on_cpu():
    """Test game that has cuda=False """

    allocation, payments = pd_cpu.play(actions_pd_cpu)

    assert allocation.device.type == 'cpu', "Result should be on CPU for cpu inputs!"
    assert payments.device.type == 'cpu', "Result should be on CPU for cpu inputs!"

    allocation, payments = pd_cpu.play(actions_pd_gpu)

    assert allocation.device.type == 'cpu', "Result should be on CPU for gpu inputs!"
    assert payments.device.type == 'cpu', "Result should be on CPU for gpu inputs!"

def test_output_shapes():    
    assert pd_allocation.shape == torch.Size([4, 2, 1]), \
        "Invalid allocation shape! Should be batch x n_players x items"
    assert pd_payments.shape == torch.Size([4, 2]), \
        "Invalid payments shape. Should be batch x n_players"

def test_output_correctness_2x2():
    assert torch.equal(pd_allocation, torch.zeros_like(pd_allocation)), \
        "All allocations in matrix game should be 0"
    assert torch.equal(pd_payments, -pd.outcomes.view(4,2)), \
        "incorrect payments. Should be negative of outcome."

def test_output_correctness_3x2():
    """3 player 2 action game: Jordan Anticoordination game."""
    jordan = JordanGame()

    actions = torch.tensor(
        [   [[0], [0], [0]], # LLL
            [[0], [1], [0]]  # LRL
        ],
        device = jordan.device)
    
    allocation, payments = jordan.play(actions)
    assert torch.equal(allocation, torch.zeros(2, 3, 1, device=jordan.device)), \
        "Found invalid allocation."

    assert torch.equal(payments, -torch.tensor(
            [   [0.,0, 0],
                [1, 1, 0]],
            device = jordan.device
        )), "Returned invalid payments!"

def test_output_correctness_2x3():
    """ 2 player 3 action game: Rock Paper Scissors"""
    rps = RockPaperScissors(cuda = True)
    
    actions = torch.tensor(
        [
            [[0], [0]], # rock = rock
            [[0], [2]], # rock > scissors
            [[1], [1]], # paper = paper
            [[2], [2]], # scissors = scissors
            [[1], [2]], # paper < scissors
        ],
        device = rps.device
    )

    allocation, payments = rps.play(actions)

    assert torch.equal(allocation, torch.zeros(5, 2, 1, device=rps.device)), \
        "Found invalid allocation."
    
    assert torch.equal(
        payments,
        -torch.tensor(
            [
                [0.,0],
                [1, -1],
                [0,0],
                [0,0],
                [-1,1]
            ],
            device = rps.device
            )
        ), "Returned invalid payments!"
    
def test_invalid_actions_float():
    actions_float = torch.tensor([
        [[.5], [.5]],
        [[0.], [1.]],
        [[.33], [.67]],
        [[1], [1.]]
    ])

    with pytest.raises(AssertionError):
        pd.play(actions_float)
        pytest.fail("Game class should fail on invalid input: Float instead of Int")

def test_invalid_actions_shape():
    actions_invalid_shape = torch.tensor([[1], [5], [3]]) 

    with pytest.raises(AssertionError):
        pd.play(actions_invalid_shape)
        pytest.fail("Game class should validate input: Invalid action profile shape!")

def test_invalid_actions_out_of_bounds():

    actions = torch.tensor([
        [[0], [0]],
        [[0], [1]],
        [[2], [0]],
        [[1], [2]]
    ])

    with pytest.raises(AssertionError):
        pd.play(actions)
        pytest.fail("Game should validate input: Action Index out of bounds!")

    actions = torch.tensor([
        [[0], [0]],
        [[-1], [1]],
        [[1], [0]],
        [[1], [2]]
    ])

    with pytest.raises(AssertionError):
        pd.play(actions)
        pytest.fail("Game should validate input: Action Index out of bounds!")

