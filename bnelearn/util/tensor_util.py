"""This module implements util functions for PyTorch tensor operations."""

import torch


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

def all_combinations(tensor: torch.Tensor) -> torch.Tensor:
    """Calculate all combinations of axis two of `tensor`, where the third axis
    is kept together (e.g. corresponding to one high dimensional point) along
    the first axis (e.g. multiple points).
    """
    batch_size, dim_a_size, dim_b_size = tensor.shape

    combinations = torch.zeros(batch_size**dim_a_size, dim_a_size, dim_b_size,
                               device=tensor.device)
    for b in range(dim_b_size):
        temp = torch.meshgrid([tensor[:, a, b] for a in range(dim_a_size)])
        for a in range(dim_a_size):
            combinations[:, a, b] = temp[a].flatten()

    return combinations
