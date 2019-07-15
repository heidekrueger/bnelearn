# -*- coding: utf-8 -*-
"""
Implementations of strategies for playing in Auctions and Matrix Games.
"""
from abc import ABC, abstractmethod
from typing import Callable, Iterable

import math

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

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

            with torch.multiprocessing.Pool(pool_size) as p:
                print('Calculating strategy for batch of size {} with a pool size of {}.'.format(inputs.shape[0],
                                                                                                 pool_size))
                result = p.map(self.closure, inputs.cpu())
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
    A strategy played by a fully connected neural network

    Args:
        input_length:
            dimension of the input layer
        hidden_nodes:
            Iterable of number of nodes in hidden layers
        hidden_activations:
            Iterable of activation functions to be used in the hidden layers.
            Should be instances of classes defined in `torch.nn.modules.activation`
        requires_grad:
            whether pytorch should build the whole DAG.
            Since ES is gradient-free, we can save some cycles and memory here.
        ensure_positive_output (optional): torch.Tensor
            When provided, will check whether the initialized model will return a
            positive bid anywhere at the given input tensor. Otherwise,
            the weights will be reinitialized.
    """
    def __init__(self, input_length: int,
                 hidden_nodes: Iterable[int],
                 hidden_activations: Iterable[nn.Module],
                 requires_grad = True,
                 ensure_positive_output: torch.Tensor or None = None):

        assert len(hidden_nodes) == len(hidden_activations), \
            "Provided nodes and activations do not match!"

        nn.Module.__init__(self)

        self.requires_grad = requires_grad
        self.input_length = input_length
        self.hidden_nodes = hidden_nodes
        self.activations = hidden_activations

        self.layers = nn.ModuleDict()

        # first layer
        self.layers['fc_0'] = nn.Linear(input_length, hidden_nodes[0])
        self.layers['activation_0'] = hidden_activations[0]

        for i in range (1, len(hidden_nodes)):
            self.layers['fc_' + str(i)] = nn.Linear(hidden_nodes[i-1], hidden_nodes[i])
            self.layers['activation_' + str(i)] = hidden_activations[i]

        self.layers['fc_out'] = nn.Linear(hidden_nodes[-1], 1)
        self.layers['activation_out'] = nn.ReLU()
        self.activations.append(nn.ReLU())

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
        self.__init__(self.input_length, self.hidden_nodes, self.activations[:-1],
                      self.requires_grad, ensure_positive_output)

    def forward(self, x):
        for layer in self.layers.values():
            x = layer(x)
        return x

    def play(self,inputs):
        return self.forward(inputs)

class TruthfulStrategy(Strategy, nn.Module):
    """A strategy that plays truthful valuations."""
    def __init__(self):
        nn.Module.__init__(self)
        self.register_parameter('dummy',nn.Parameter(torch.zeros(1)))

    def forward(self, x):
        return x

    def play(self, inputs):
        return self.forward(inputs)
