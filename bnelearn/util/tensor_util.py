"""This module implements util functions for PyTorch tensor operations."""

import traceback
from tqdm import tqdm
from typing import List
from math import ceil
import torch

_CUDA_OOM_ERR_MSG_START = "CUDA out of memory. Tried to allocate"
ERR_MSG_OOM_SINGLE_BATCH = "Failed for good. Even batch_size=1 leads to OOM!"


def batched_index_select(input: torch.Tensor, dim: int,
                         index: torch.Tensor) -> torch.Tensor:
    """
    Extends the torch ´index_select´ function to be used for multiple batches
    at once.

    This code is borrowed from https://discuss.pytorch.org/t/batched-index-select/9115/11.

    author:
        dashesy

    args:
        input: Tensor which is to be indexed
        dim: Dimension
        index: Index tensor which proviedes the seleting and ordering.

    returns:
        Indexed tensor
    """
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)

    return torch.gather(input, dim, index)

def iterative_evaluation(function: callable, args: torch.Tensor, n_outputs: int=1,
                         dtypes: List[str]=torch.float):
    """Apply the function `function` to the arguments `args`. Split up
    calculations into chunks over dimension `0` when running out of memory.

    Args:
        function :callable: function to be evaluated.
        args :torch.Tensor: pytorch.tensor arguments passed to function.
        n_outputs :int: the number of flat tensors to be returned.
        dtypes: :List[str]: the dtypes of the outputs. Length must match length
            of `n_outputs`.

    Returns:
        function evaluated at args.
    """
    mini_batch_size = args.shape[0]
    calculation_successful = False
    output = [torch.empty(mini_batch_size, dtype=dtypes[i], device=args.device)
              for i in range(n_outputs)]

    while not calculation_successful:
        try:
            print(f"Trying {function} calculation with batch_size {mini_batch_size}...")
            mini_args = args.split(mini_batch_size)
            for i, mini_arg in tqdm(enumerate(mini_args), total=ceil(len(mini_args))):

                # get the indices corresponding to this mini batch
                indices = slice(i*mini_batch_size, (i+1)*mini_batch_size)

                out = function(mini_arg)
                for out_i in range(n_outputs):
                    output[out_i][indices] = out[out_i]

            calculation_successful = True
            print("\t ... success!")

        except RuntimeError as e:
            if not str(e).startswith(_CUDA_OOM_ERR_MSG_START):
                raise e
            if mini_batch_size <= 1:
                traceback.print_exc()
                # pylint: disable = raise-missing-from
                raise RuntimeError(ERR_MSG_OOM_SINGLE_BATCH) 

            print("\t... failed (OOM). Decreasing mini batch size.")
            mini_batch_size = int(mini_batch_size / 2)

    return output
