"""Testing correctness of reverse auction implementations."""

import pytest
import torch

from bnelearn.mechanism import FPSBSplitAwardAuction

cuda = torch.cuda.is_available()
fpsbsaa = FPSBSplitAwardAuction(cuda=cuda)

fpsbsaa_bids = torch.tensor([
    [[3.00, 1.00],
     [3.70, 1.00],
     [3.60, 1.50]
    ],
    [[2.00, 0.50],
     [1.00, 1.50],
     [0.99, 0.49]
    ],
    [[2.00, 1.00],
     [2.50, 1.00],
     [1.50, 1.50]
    ]], device = device)

fpsbsaa_bids_allocations = torch.tensor([
    [[0., 1.],
     [0., 1.],
     [0., 0.]
    ],
    [[0., 1.],
     [0., 0.],
     [0., 1.]
    ],
    [[0., 0.],
     [0., 0.],
     [1., 0.]
    ]], device = device)

def test_fpsbsaa_correctness():
    """Test of allocation and payments in First-price sealed-bid split-award auction."""

    # test 0
    allocations, payments = fpsbsaa.run(fpsbsaa_bids)
    assert torch.equal(allocations, fpsbsaa_bids_allocations)
    assert torch.equal(payments, torch.tensor(
        [[1.0000, 1.0000, 0.0000],
         [0.5000, 0.0000, 0.4900],
         [0.0000, 0.0000, 1.5000]],
        device = payments.device))
