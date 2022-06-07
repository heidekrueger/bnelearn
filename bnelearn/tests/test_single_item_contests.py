"""Testing correctness of contest implementations."""

import pytest
import torch

from bnelearn.mechanism import TullockContest, CrowdsourcingContest

cuda = torch.cuda.is_available()

# Contest specific variables
tullock_impact_factor = 1
impact_function = lambda x: x ** tullock_impact_factor

tullock = TullockContest(cuda=cuda, impact_function=impact_function)

crowdsourcing = CrowdsourcingContest(cuda=cuda)

device = tullock.device

bids = torch.tensor([
    [[0.5], [2], [1.5]],
    [[3], [0.5], [1.5]],
    [[0], [4], [1]]
])

bids_cpu = bids.cpu()

bids_illegal_negative = torch.tensor([
    [[1.5], [2.3], [3.3]],
    [[-1.6], [1.8], [1.3]],
    [[1], [2], [3]]
], device=device)

bids_illegal_dimensions = torch.tensor([
    [1, 2, 3]
], device= device)

def test_tullock_cuda():
    """Tullock Contest should run on GPU if available on the system and desired."""
    if not torch.cuda.is_available():
        pytest.skip("This test requires CUDA, but it's not available.")

    bids_gpu = bids_cpu.cuda()

    allocations, payments = tullock.run(bids_cpu)
    allocations1, payments1 = tullock.run(bids_gpu)

    assert all(
        [tensor.device.type == 'cuda'
        for tensor in [allocations, allocations1, payments, payments1]
        ]), 'Outputs should be on gpu!'

def test_crowdsourcing_cuda():
    """Tullock Contest should run on GPU if available on the system and desired."""
    if not torch.cuda.is_available():
        pytest.skip("This test requires CUDA, but it's not available.")

    bids_gpu = bids_cpu.cuda()

    allocations, payments = crowdsourcing.run(bids_cpu)
    allocations1, payments1 = crowdsourcing.run(bids_gpu)

    assert all(
        [tensor.device.type == 'cuda'
        for tensor in [allocations, allocations1, payments, payments1]
        ]), 'Outputs should be on gpu!'

def test_tullock_illegal_arguments():
    """Illegal effort tensors should cause exceptions."""
    with pytest.raises(AssertionError):
        tullock.run(bids_illegal_negative)

    with pytest.raises(AssertionError):
        tullock.run(bids_illegal_dimensions)

def test_crowdsourcing_illegal_arguments():
    """Illegal effort tensors should cause exceptions."""
    with pytest.raises(AssertionError):
        crowdsourcing.run(bids_illegal_negative)

    with pytest.raises(AssertionError):
        crowdsourcing.run(bids_illegal_dimensions)

def test_tullock_correctness():
    """Tullock Contest should return correct winning probabilities and payments."""

    winning_probs, payments = tullock.run(bids)

    assert torch.equal(winning_probs, torch.tensor(
        [
            [[0.125], [0.5], [0.375]],
            [[0.6], [0.1], [0.3]],
            [[0], [0.8], [0.2]]
        ], device=winning_probs.device
    ))

    assert torch.equal(payments, torch.tensor(
        [
            [0.5, 2, 1.5],
            [3, 0.5, 1.5],
            [0, 4, 1]
        ], device=payments.device
    ))

def test_crowdsourcing_correctness():
    """Crowdsourcing Contest should return correct allocations and payments."""

    allocations, payments = crowdsourcing.run(bids)

    assert torch.equal(allocations, torch.tensor(
        [
            [[2], [0], [1]],
            [[0], [2], [1]],
            [[2], [0], [1]]
        ], device=allocations.device
    ))

    assert torch.equal(payments, torch.tensor(
        [
            [0.5, 2, 1.5],
            [3, 0.5, 1.5],
            [0, 4, 1]
        ], device=payments.device
    ))


