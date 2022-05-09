"""This module implements util functions for PyTorch tensor operations."""

import traceback
from tqdm import tqdm
from typing import List
from math import ceil
import torch

import torch.nn as nn

_CUDA_OOM_ERR_MSG_START = "CUDA out of memory. Tried to allocate"
ERR_MSG_OOM_SINGLE_BATCH = "Failed for good. Even a batch_size of 1 leads to OOM!"


class GaussLayer(nn.Module):
    """
    Custom layer for normally distributed predictions (non-negative).

    Has no trainable parameters.
    """
    def __init__(self, **kwargs):
        super(GaussLayer, self).__init__(**kwargs)
        self.mixed_strategy = True
        self.log_prob = None

    # pylint: disable=fixme, missing-function-docstring
    def forward(self, x, deterministic=False, pretrain=False):
        if x.dim() == 1:
            x = x.view(-1, 1)
        m = x.shape[-1] // 2

        # return mean actions
        if deterministic:
            return x[..., :m]

        mean = x[..., :m]
        std = x[..., m:].exp()

        normal = torch.distributions.normal.Normal(mean, std)

        # Pretrain is supervised learning -> `rsample` is differentable,
        # otherwise we differentiate though the log probabilites
        if pretrain:
            return normal.rsample()
        else:
            out = normal.sample()

        if self.training:
            self.log_prob = normal.log_prob(out)

        return out


class UniformLayer(nn.Module):
    """
    Custom layer for predictions following a uniform distribution.

    Has no trainable parameters.
    """
    def __init__(self, **kwargs):
        super(UniformLayer, self).__init__(**kwargs)
        self.mixed_strategy = True
        self.log_prob = None

    # pylint: disable=fixme, missing-function-docstring
    def forward(self, x, deterministic=False, pretrain=False):
        if x.dim() == 1:
            x = x.view(-1, 1)
        m = x.shape[-1] // 2

        # return mean actions
        if deterministic:
            return x[..., :m]

        uniform = torch.distributions.uniform.Uniform(x[..., :m], x[..., :m] + x[..., m:].exp())

        if pretrain:
            return uniform.rsample()
        else:
            out = uniform.sample()

        if self.training:
            self.log_prob = uniform.log_prob(out)

        return out


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


def apply_with_dynamic_mini_batching(
        function: callable,
        args: torch.Tensor,
        n_outputs: int=1,
        dtypes: List[type]=[torch.float]
    ) -> List[torch.Tensor]:
    """Apply the function `function` batch wise to the tensor argument `args`
    with error handling for CUDA Out-Of-Memory problems. Starting with the full
    batch, this method will cut the batch size in half until the operation
    suceeds (or a non-CUDA-OOM error occurs).

    NOTE: The automatic error handling applies to CUDA memory limits only. This
    function does not provide any benefits when processing on CPU with regular
    RAM.

    Args:
        function :callable: function to be evaluated.
        args :torch.Tensor: pytorch.tensor arguments passed to function.
        n_outputs :int: the number of flat tensors to be returned.
        dtypes: :List[type]: the dtypes of the outputs. Length must match length
            of `n_outputs`.

    Returns:
        function evaluated at args.
    """
    batch_size = args.shape[0]
    output = [torch.empty(batch_size, dtype=dtypes[i], device=args.device)
              for i in range(n_outputs)]

    calculation_successful = False
    mini_batch_size = batch_size

    while not calculation_successful:
        try:
            print(f"Trying {function} calculation with batch_size {mini_batch_size}...")

            # Split up arguments into smaller chunks of batch size `mini_batch_size`
            mini_args = args.split(mini_batch_size)

            # Iterate over chunks
            for i, mini_arg in tqdm(enumerate(mini_args), total=ceil(len(mini_args))):

                # Get the indices corresponding to this mini batch
                indices = slice(i*mini_batch_size, (i+1)*mini_batch_size)

                mini_output = function(mini_arg)
                for out_dim in range(n_outputs):
                    output[out_dim][indices] = mini_output[out_dim]

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
