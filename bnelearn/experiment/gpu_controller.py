import warnings
import torch


class GPUController:
    def __init__(self, cuda: bool = True, specific_gpu: int = 0):
        self.cuda = cuda
        self.specific_gpu = specific_gpu

        if cuda and not torch.cuda.is_available():
            warnings.warn('Cuda not available. Falling back to CPU!')
            cuda = False
        self.device = 'cuda' if cuda else 'cpu'

        if cuda and specific_gpu:
            torch.cuda.set_device(specific_gpu)

