"""Testing correctness of auction implementations."""

import pytest
import torch

from bnelearn.mechanism import FirstPriceSealedBidAuction, VickreyAuction, ThirdPriceSealedBidAuction, AllPayAuction

cuda = torch.cuda.is_available()

fpsb = FirstPriceSealedBidAuction(cuda=cuda)
vickrey = VickreyAuction(cuda=cuda)
third = ThirdPriceSealedBidAuction(cuda=cuda)
allpay = AllPayAuction(cuda=cuda)

device = fpsb.device

bids_unambiguous = torch.tensor([
    [[1,   2.1,    3],
     [3.7, 2,    0],
     [3.6, 1.99, 2.99]
    ],
    [[1.0, 1.0, 1.01],
     [1.01, .99, 1.0],
     [1.0, .99, 1.0]
    ]], device = device)

bids_ambiguous = torch.tensor([
    [[1,   2,    3],
     [3.7, 2,    0],
     [3.6, 1.99, 2.99]
    ],
    [[1.0, 1.0, 1.0],
     [1.0, 1.0, 1.0],
     [1.0, 1.0, 1.0]
    ]], device = device)
bids_cpu = bids_ambiguous.cpu()

bids_illegal_negative = torch.tensor([
    [[1,   2,    3],
     [3.7, 2,    0],
     [3.6, 1.99, 2.99]
    ],
    [[1.0, 1.0, 1.0],
     [1.0, 1.0, -1.0],
     [1.0, 1.0, 1.0]
    ]], device = device)

bids_illegal_dimensions = torch.tensor([
    [1, 2, 3]
    ], device = device)

bids_allpay = torch.tensor([
    [[1.0], [2.0], [3.0]],
    [[0.5], [1.5], [0.25]],
    [[2.0], [1.0], [0.0]],
])

def test_fpsb_illegal_arguments():
    """Illegal bid tensors should cause exceptions"""
    with pytest.raises(AssertionError):
        fpsb.run(bids_illegal_negative)

    with pytest.raises(AssertionError):
        fpsb.run(bids_illegal_dimensions)

def test_fpsb_correctness():
    """FPSB should return correct allocations and payments."""

    allocations, payments = fpsb.run(bids_unambiguous)

    assert torch.equal(allocations, torch.tensor(
        [
            [[0., 1., 1.],
             [1., 0., 0.],
             [0., 0., 0.]],
            [[0., 1., 1.],
             [1., 0., 0.],
             [0., 0., 0.]]
        ], device = allocations.device))

    assert torch.equal(payments, torch.tensor(
        [[5.1000, 3.7000, 0.0000],
         [2.0100, 1.0100, 0.0000]],
        device = payments.device))

def test_vickrey_correctness():
    """Vickrey should return correct allocations and payments."""

    allocations, payments = vickrey.run(bids_unambiguous)

    assert torch.equal(allocations, torch.tensor(
        [
            [[0., 1., 1.],
             [1., 0., 0.],
             [0., 0., 0.]],
            [[0., 1., 1.],
             [1., 0., 0.],
             [0., 0., 0.]]
        ], device = allocations.device))

    assert torch.equal(payments, torch.tensor(
        [[4.9900, 3.6000, 0.0000],
         [1.9900, 1.0000, 0.0000]],
        device = payments.device))

def test_thirdprice_illegal_arguments():
    """Illegal bid tensors should cause exceptions"""
    with pytest.raises(AssertionError):
        third.run(bids_unambiguous)

def test_thirdprice_correctness():
    """Thirdprice should return correct allocations and payments."""
    bids_unambiguous[0,1,2] = 0.1
    allocations, payments = third.run(bids_unambiguous)
    assert torch.equal(allocations, torch.tensor(
        [
            [[0., 1., 1.],
             [1., 0., 0.],
             [0., 0., 0.]],
            [[0., 1., 1.],
             [1., 0., 0.],
             [0., 0., 0.]]
        ], device = allocations.device))

    assert torch.equal(payments, torch.tensor(
        [[2.09, 1.0, 0.0000],
         [1.99, 1.0, 0.0000]],
        device = payments.device))

def test_allpay_illegal_arguments():
    """Illegal bid tensors should cause exceptions"""
    with pytest.raises(AssertionError):
        allpay.run(bids_illegal_negative)

    with pytest.raises(AssertionError):
        allpay.run(bids_illegal_dimensions)

def test_allpay_correctness():
    """Thirdprice should return correct allocations and payments."""
    bids_unambiguous[0,1,2] = 0.1
    allocations, payments = allpay.run(bids_allpay)
    assert torch.equal(allocations, torch.tensor(
        [
            [[0.], [0.], [1.]],
            [[0.], [1.], [0.]],
            [[1.], [0.], [0.]]
        ], device = allocations.device))

    assert torch.equal(payments, torch.tensor(
        [[1.0, 2.0, 3.0],
         [0.5, 1.5, 0.25],
         [2.0, 1.0, 0.0]],
        device = payments.device))