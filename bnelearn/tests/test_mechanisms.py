import warnings
import pytest
import torch
import bnelearn.mechanism


def test_dummy():
    assert 1==1 

def test_fpsb_cuda():
    
    if not torch.cuda.is_available():
        pytest.skip("This test needs CUDA, but it's not available.")
    
    from bnelearn.mechanism import FirstPriceSealedBidAuction

    fpsb = FirstPriceSealedBidAuction(cuda=True)
    
    bids_cpu = torch.Tensor([
        [[1,   2,    3],
         [3.7, 2,    0],
         [3.6, 1.99, 2.99]
        ],
        [[1.0, 1.0, 1.0],
         [1.0, 1.0, 1.0],
         [1.0, 1.0, 1.0]            
        ]])
    bids_gpu = bids_cpu.cuda()

    allocations, payments = fpsb.run(bids_cpu)
    allocations1, payments1 = fpsb.run(bids_gpu)

    assert all(
        [tensor.device.type == 'cuda'
         for tensor in [allocations, allocations1, payments, payments1]
        ]), "outputs should be on gpu!"

if __name__ == "__main__":
    print('hello.')