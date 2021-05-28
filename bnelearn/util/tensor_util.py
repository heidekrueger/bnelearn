"""This module implements util functions for PyTorch tensor operations."""

import torch


def batched_index_select(input: torch.Tensor, dim: int,
                         index: torch.Tensor) -> torch.Tensor:
    """
    Extends the torch ´index_select´ function to be used for multiple batches
    at once.

    author:
        dashesy @ https://discuss.pytorch.org/t/batched-index-select/9115/11

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
