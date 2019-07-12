# -*- coding: utf-8 -*-
"""
Implementations of strategies for playing in Auctions and Matrix Games.
"""
from abc import ABC, abstractmethod
from typing import Callable

import math

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.multiprocessing import Pool, set_sharing_strategy

## E1102: false positive on torch.tensor()
## false positive 'arguments-differ' warnings for forward() overrides
# pylint: disable=arguments-differ

class Strategy(ABC):
    """A Strategy to map (optional) private inputs of a player to actions in a game."""

    @abstractmethod
    def play(self, inputs):
        """Takes (private) information as input and decides on the actions an agent should play."""
        raise NotImplementedError()

class ClosureStragegy(Strategy):
    """A strategy specified by a closure"""

    def __init__(self, closure: Callable, parallel: int = 0):
        self.closure = closure
        self.parallel = parallel

    def play(self, inputs):
        pool_size = 1

        if self.parallel:
            pool_size = min(self.parallel, max(1, math.ceil(inputs.shape[0]/2**10)))

        if pool_size > 1:
            in_device = inputs.device

            # calculate necessary shape by calling closure for a single input
            _, *other_dims = self.closure(inputs[:1]).shape
            out_shape = torch.Size([inputs.shape[0], *other_dims])

            torch.multiprocessing.set_sharing_strategy('file_system')

            with Pool(pool_size) as p:
                print('Calculating strategy for batch of size {} with a pool size of {}.'.format(inputs.shape[0],
                                                                                                 pool_size))
                result = p.map(self.closure, inputs.cpu())
                # TODO: only implemented for 1-dimensional bids -- otherwise shape might be incorrect.
                result = torch.tensor(result, device = in_device).view(out_shape)

            return result

        return self.closure(inputs)


class MatrixGameStrategy(Strategy, nn.Module):
    """ A dummy neural network that encodes and returns a mixed strategy"""
    def __init__(self, n_actions):
        nn.Module.__init__(self)
        self.logits = nn.Linear(1, n_actions)

        for param in self.parameters():
            param.requires_grad = False

        # initialize distribution
        self._update_distribution()

    def _update_distribution(self):
        self.device = next(self.parameters()).device
        probs = self.forward(torch.ones(1,  device=self.device))
        self.distribution = Categorical(probs=probs)

    def forward(self, x):
        logits = self.logits(x)
        probs = torch.softmax(logits, 0)
        return probs

    def play(self, inputs=None, batch_size = 1):
        if inputs is None:
            inputs= torch.ones(batch_size, 1, device=self.device)

        self._update_distribution()
        # is of shape batch size x 1
        # TODO: this is probably slow AF. fix when needed.
        return self.distribution.sample(inputs.shape)

class NeuralNetStrategy(Strategy, nn.Module):
    """
    A strategy played by a neural network


    Args:
        input_length: dimension of the input layer
        size_hidden_layer: number of nodes in (currently single) hidden layer
        requires_grad:
            whether pytorch should build the whole DAG.
            Since ES is gradient-free, we can save some cycles and memory here.
        ensure_positive_output (optional): torch.Tensor
            When provided, will check whether the initialized model will return a
            positive bid anywhere at the given input tensor. Otherwise,
            the weights will be reinitialized.
    """
    def __init__(self, input_length, size_hidden_layer = 10, requires_grad = True,
                 ensure_positive_output: torch.Tensor or None = None):
        nn.Module.__init__(self)

        self.requires_grad = requires_grad
        self.input_length = input_length
        self.size_hidden_layer = size_hidden_layer
        self.fc1 = nn.Linear(input_length, size_hidden_layer)
        #self.fc2 = nn.Linear(size_hidden_layer, size_hidden_layer)
        self.fc_out = nn.Linear(size_hidden_layer, 1)
        #self.tanh = nn.Tanh()
        #self.lrelu1 = nn.LeakyReLU(negative_slope=.1)
        #self.lrelu2 = nn.LeakyReLU(negative_slope=.1)

        # turn off gradients if not required (e.g. for ES-training)
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        # test whether output at ensure_positive_output is positive,
        # if it isn't --> reset the initialization
        if ensure_positive_output:
            if not any(self.forward(ensure_positive_output).gt(0)):
                self.reset(ensure_positive_output)

    def reset(self, ensure_positive_output=None):
        """Re-initialize weights of the Neural Net, ensuring positive model output for a given input."""
        self.__init__(self.input_length, self.size_hidden_layer, self.requires_grad, ensure_positive_output)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        #x = F.tanh(self.fc2(x))
        #x = self.tanh(self.fc1(x))
        x = self.fc_out(x).relu()

        return x

    def play(self,inputs):
        return self.forward(inputs)

class TruthfulStrategy(Strategy, nn.Module):
    def __init__(self, input_length):
        nn.Module.__init__(self)
        self.register_parameter('dummy',nn.Parameter(torch.zeros(1)))

    def forward(self, x):
        # simply play first input
        # TODO: right now specific for input length 2!
        return x.matmul(torch.tensor([[1.0], [0.0]], device=x.device))

    def play(self, inputs):
        return self.forward(inputs)
