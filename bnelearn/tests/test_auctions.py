"""Testing correctness of auction implementations."""

import pytest
import torch
from bnelearn.mechanism import FirstPriceSealedBidAuction, VickreyAuction, \
    MultiItemDiscriminatoryAuction, MultiItemUniformPriceAuction, \
    MultiItemVickreyAuction

cuda = True

fpsb = FirstPriceSealedBidAuction(cuda=cuda)
vickrey = VickreyAuction(cuda=cuda)
mida = MultiItemDiscriminatoryAuction(cuda=cuda)
miup = MultiItemUniformPriceAuction(cuda=cuda)
miva = MultiItemVickreyAuction(cuda=cuda)

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

bids_multi_unit_0 = torch.tensor([
    [[3.00, 2.00, 1.00],
     [3.70, 2.00, 0.00],
     [3.60, 2.99, 1.99]
    ],
    [[1.02, 1.01, 1.00],
     [1.01, 0.99, 0.93],
     [0.99, 0.95, 0.93]
    ],
    [[4.0, 3.0, 2.0],
     [1.0, 0.9, 0.0],
     [1.5, 1.1, 1.0]
    ]], device = device)

bids_multi_unit_0_allocations = torch.tensor([
    [[1., 0., 0.],
     [1., 0., 0.],
     [1., 0., 0.]
    ],
    [[1., 1., 0.],
     [1., 0., 0.],
     [0., 0., 0.]
    ],
    [[1., 1., 1.],
     [0., 0., 0.],
     [0., 0., 0.]
    ]], device = device)

bids_multi_unit_1 = torch.tensor([
    [[3.00, 2.00, 1.00, 1.00],
     [3.70, 2.00, 0.00, 0.00],
     [3.60, 2.99, 1.99, 1.00]
    ],
    [[1.02, 1.01, 1.00, 0.50],
     [1.01, 0.99, 0.93, 0.00],
     [0.99, 0.95, 0.93, 0.50]
    ]], device = device)

bids_multi_unit_1_allocations = torch.tensor([
    [[1., 0., 0., 0.],
     [1., 0., 0., 0.],
     [1., 1., 0., 0.]
    ],
    [[1., 1., 1., 0.],
     [1., 0., 0., 0.],
     [0., 0., 0., 0.]
    ]], device = device)

def test_fpsb_cuda():
    """FPSB should run on GPU if available on the system and desired."""

    if not torch.cuda.is_available():
        pytest.skip("This test needs CUDA, but it's not available.")

    bids_gpu = bids_cpu.cuda()

    allocations, payments = fpsb.run(bids_cpu)
    allocations1, payments1 = fpsb.run(bids_gpu)

    assert all(
        [tensor.device.type == 'cuda'
         for tensor in [allocations, allocations1, payments, payments1]
        ]), "Outputs should be on gpu!"

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

def test_mida_correctness():
    """Test of allocation and payments in MultiItemDiscriminatoryAuction."""

    # test 0
    allocations, payments = mida.run(bids_multi_unit_0)
    assert torch.equal(allocations, bids_multi_unit_0_allocations)
    assert torch.equal(payments, torch.tensor(
        [[3.0000, 3.7000, 3.6000],
         [2.0300, 1.0100, 0.0000],
         [9.0000, 0.0000, 0.0000]],
        device = payments.device))

    # test 1
    allocations, payments = mida.run(bids_multi_unit_1)
    assert torch.equal(allocations, bids_multi_unit_1_allocations)
    assert torch.equal(payments, torch.tensor(
        [[3.0000, 3.7000, 6.5900],
         [3.0300, 1.0100, 0.0000]],
        device = payments.device))

def test_miup_correctness():
    """Test of allocation and payments in MultiItemUniformPriceAuction."""

    # test 0
    allocations, payments = miup.run(bids_multi_unit_0)
    assert torch.equal(allocations, bids_multi_unit_0_allocations)
    assert torch.equal(payments, torch.tensor(
        [[2.9900, 2.9900, 2.9900],
         [2.0000, 1.0000, 0.0000],
         [4.5000, 0.0000, 0.0000]],
        device = payments.device))

    # test 1
    allocations, payments = miup.run(bids_multi_unit_1)
    assert torch.equal(allocations, bids_multi_unit_1_allocations)
    assert torch.equal(payments, torch.tensor(
        [[2.0000, 2.0000, 4.0000],
         [2.9700, 0.9900, 0.0000]],
        device = payments.device))

def test_miva_correctness():
    """Test of allocation and payments in MultiItemVickreyAuction."""

    # test 0
    allocations, payments = miva.run(bids_multi_unit_0)
    assert torch.equal(allocations, bids_multi_unit_0_allocations)
    assert torch.equal(payments, torch.tensor(
        [[2.9900, 2.9900, 2.0000],
         [1.9800, 1.0000, 0.0000],
         [3.6000, 0.0000, 0.0000]],
        device = payments.device))

    # test 1
    allocations, payments = miva.run(bids_multi_unit_1)
    assert torch.equal(allocations, bids_multi_unit_1_allocations)
    assert torch.equal(payments, torch.tensor(
        [[2.0000, 2.0000, 4.0000],
         [2.9300, 0.9900, 0.0000]],
        device = payments.device))
