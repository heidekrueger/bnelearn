# -*- coding: utf-8 -*-
"""
Implementations of strategies for playing in Auctions and Matrix Games.
"""
from abc import ABC, abstractmethod
from typing import Callable

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

## E1102: false positive on torch.tensor()
## false positive 'arguments-differ' warnings for forward() overrides
# pylint: disable=arguments-differ

class Strategy(ABC):

    @abstractmethod
    def play(self, inputs):
        """Takes (private) information as input and decides on the actions an agent should play."""
        raise NotImplementedError()

class ClosureStragegy(Strategy):
    """A strategy specified by a closure"""

    def __init__(self, closure: Callable):
        self.closure = closure

    def play(self, inputs):
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
    """ A strategy played by a neural network"""
    def __init__(self, input_length, size_hidden_layer = 10, requires_grad = True):
        nn.Module.__init__(self)
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
