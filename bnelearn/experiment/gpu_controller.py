import warnings
import torch


class GPUController:
    def __init__(self, cuda: bool = True, specific_gpu: int = None):
        self.cuda = cuda
        self.specific_gpu = specific_gpu
        self.fallback = False

        if cuda and not torch.cuda.is_available():
            warnings.warn('Cuda not available. Falling back to CPU!')
            self.cuda = False
            self.fallback = True
        self.device = 'cuda' if cuda else 'cpu'

        if cuda and specific_gpu is not None:
            torch.cuda.set_device(specific_gpu)
