# -*- coding: utf-8 -*-
"""
Implementations of strategies for playing in Auctions and Matrix Games.
"""
import math
from abc import ABC, abstractmethod
from copy import copy
from typing import Callable, Iterable
import os
import sys
import warnings

import torch
import torch.nn as nn
from tqdm import tqdm

## E1102: false positive on torch.tensor()
## false positive 'arguments-differ' warnings for forward() overrides
# pylint: disable=arguments-differ

class Strategy(ABC):
    """A Strategy to map (optional) private inputs of a player to actions in a game."""

    @abstractmethod
    def play(self, inputs):
        """Takes (private) information as input and decides on the actions an agent should play."""
        raise NotImplementedError()

    def pretrain(self, input_tensor, iterations, transformation=None):
        """If implemented by subclass, pretrains the strategy to yield desired initial outputs."""
        # pylint: disable=unused-argument # this method is 'soft-abstract'
        warnings.warn('Strategy of type {} does not support pretraining'.format(str(type(self))))

class ClosureStrategy(Strategy):
    """A strategy specified by a closure

        Args:
            closure: Callable a function or lambda that defines the strategy
            parallel: int (optional) maximum number of processes for parallel execution of closure. Default is
                      0/1 (i.e. no parallelism)
    """


    def __init__(self, closure: Callable, parallel: int = 0, mute=False):
        if not isinstance(closure, Callable):
            raise ValueError("Provided closure must be Callable!")
        self.closure = closure
        self.parallel = parallel
        self._mute = mute

    def __mute(self):
        """suppresses stderr output from workers (avoid integration warnings for each process)"""
        if self._mute:
            sys.stderr = open(os.devnull, 'w')

    def play(self, inputs):
        pool_size = 1

        if self.parallel:
            # detect appropriate pool size
            pool_size = min(self.parallel, max(1, math.ceil(inputs.shape[0]/2**10)))

        # parallel version
        if pool_size > 1:
            in_device = inputs.device

            # calculate necessary shape by calling closure once for a single input
            _, *other_dims = self.closure(inputs[:1]).shape
            out_shape = torch.Size([inputs.shape[0], *other_dims])

            # determine chunk-size -----
            # if providing the tensor by itself, pool.map will iterate over individual elements
            # and just communicate multiple of those elements to a worker at once.
            # so instead, we'll split the tensor into a list of tensors ourselves and provide that
            # as the iterator.
            # we'll use the same chunk-size heuristic as in python.multiprocessing
            # see https://stackoverflow.com/questions/53751050
            chunksize, extra = divmod(inputs.shape[0], pool_size*4)
            if extra:
                chunksize += 1

            # move input to cpu and split into chunks
            split_tensor = inputs.cpu().split(chunksize)
            n_chunks = len(split_tensor)

            #torch.multiprocessing.set_sharing_strategy('file_system') # needed for very large number of chunks

            with torch.multiprocessing.Pool(pool_size, initializer=self.__mute) as p:
                # as we handled chunks ourselves, each element of our list should be an individual chunk,
                # so the pool.map will get argument chunksize=1
                # The following code is wrapped to produce progess bar, without it simplifies to:
                # result = p.map(self.closure, split_tensor, chunksize=1)
                result = list(tqdm(
                    p.imap(self.closure, split_tensor, chunksize=1),
                    total = n_chunks, unit='chunks',
                    desc = 'Calculating strategy for batch_size {} with {} processes, chunk size of {}'.format(
                        inputs.shape[0], pool_size, chunksize)
                    ))

            # finally stitch the tensor back together
            result = torch.cat(result).view(out_shape).to(in_device)
            return result

        # serial version on single processor
        return self.closure(inputs)

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
        ensure_positive_output (optional): torch.Tensor
            When provided, will check whether the initialized model will return a
            positive bid anywhere at the given input tensor. Otherwise,
            the weights will be reinitialized.
        output_length (optional): int
            length of output/action vectorm defaults to 1
            (currently given last for backwards-compatibility)
        dropout (optional): float
            If not, applies AlphaDropout (https://pytorch.org/docs/stable/nn.html#torch.nn.AlphaDropout)
            to `dropout` share of nodes in each hidden layer during training.

    """
    def __init__(self, input_length: int,
                 hidden_nodes: Iterable[int],
                 hidden_activations: Iterable[nn.Module],
                 ensure_positive_output: torch.Tensor or None = None,
                 output_length: int = 1, # currently last argument for backwards-compatibility
                 dropout: float = 0.0
                 ):

        assert len(hidden_nodes) == len(hidden_activations), \
            "Provided nodes and activations do not match!"

        nn.Module.__init__(self)

        self.input_length = input_length
        self.output_length = output_length
        self.hidden_nodes = copy(hidden_nodes)
        self.activations = copy(hidden_activations) # do not write to list outside!
        self.dropout = dropout

        self.layers = nn.ModuleDict()

        if len(hidden_nodes) > 0:
            ## create hdiden layers
            # first hidden layer (from input)
            self.layers['fc_0'] = nn.Linear(input_length, hidden_nodes[0])
            self.layers[str(self.activations[0]) + '_0'] = self.activations[0]
            if self.dropout:
                self.layers['dropout_0'] = nn.AlphaDropout(p=self.dropout)
            # hidden-to-hidden-layers
            for i in range (1, len(hidden_nodes)):
                self.layers['fc_' + str(i)] = nn.Linear(hidden_nodes[i-1], hidden_nodes[i])
                self.layers[str(self.activations[i]) + '_' + str(i)] = self.activations[i]
                if self.dropout:
                    self.layers['dropout_' + str(i)] = nn.AlphaDropout(p=self.dropout)
        else:
            # output layer directly from inputs
            hidden_nodes = [input_length] #don't write to self.hidden nodes, just ensure correct creation

        # create output layer
        self.layers['fc_out'] = nn.Linear(hidden_nodes[-1], output_length)
        self.layers[str(nn.ReLU()) + '_out'] = nn.ReLU()
        self.activations.append(self.layers[str(nn.ReLU()) + '_out'])

        # test whether output at ensure_positive_output is positive,
        # if it isn't --> reset the initialization
        if ensure_positive_output is not None:
            if not torch.all(self.forward(ensure_positive_output).gt(0)):
                self.reset(ensure_positive_output)

    @classmethod
    def load(cls, path: str, device='cpu'):
        """
        Initializes a saved NeuralNetStrategy from ´path´.
        """

        model_dict = torch.load(path, map_location=device)

        # TODO: Dangerous hack for reloading a startegy
        params = {}
        params["hidden_nodes"] = []
        params["hidden_activations"] = []
        length = len(list(model_dict.values()))
        layer_idx = 0
        value_key_zip = zip(
            list(model_dict.values()),
            list(model_dict._metadata.keys())[2:] # pylint: disable=protected-access
        )
        for tensor, layer_activation in value_key_zip:
            if layer_idx == 0:
                params["input_length"] = tensor.shape[1]
            elif layer_idx == length - 1:
                params["output_length"] = tensor.shape[0]
            elif layer_idx % 2 == 1:
                params["hidden_nodes"].append(tensor.shape[0])
                params["hidden_activations"].append(
                    # TODO Nils: change once models are saved correctly
                    # eval('nn.' + layer_activation[7:-2]))
                    nn.SELU())
            layer_idx += 1

        # standard initialization
        strategy = cls(
            input_length=params["input_length"],
            hidden_nodes=params["hidden_nodes"],
            hidden_activations=params["hidden_activations"],
            output_length=params["output_length"]
        )

        # override model weights with saved ones
        strategy.load_state_dict(model_dict)

        return strategy

    def pretrain(self, input_tensor: torch.Tensor, iters: int, transformation: Callable = None):
        """Performs `iters` steps of supervised learning on `input` tensor,
           in order to find an initial bid function that is suitable for learning.

           args:
               input: torch.Tensor, same dimension as self.input_length
               iters: number of iterations for supervised learning
               transformation (optional): Callable. Defaulting to identity function if input_length == output_length
           returns: Nothing
        """

        desired_output = input_tensor
        if transformation is not None:
            desired_output = transformation(input_tensor)

        if desired_output.shape[-1] < self.output_length:
            # TODO: not appropriate for CAs
            torch.cat([desired_output] * self.output_length, axis=1)
        elif desired_output.shape[-1] > self.output_length:
            raise ValueError('Desired pretraining output does not match NN output dimension.')

        optimizer = torch.optim.Adam(self.parameters())
        for _ in range(iters):
            self.zero_grad()
            diff = (self.forward(input_tensor) - desired_output)
            loss = (diff * diff).sum()
            loss.backward()
            optimizer.step()

    def reset(self, ensure_positive_output=None):
        """Re-initialize weights of the Neural Net, ensuring positive model output for a given input."""
        self.__init__(self.input_length, self.hidden_nodes,
                      self.activations[:-1], ensure_positive_output, self.output_length)

    def forward(self, x):
        for layer in self.layers.values():
            x = layer(x)
        return x

    def play(self, inputs):
        return self.forward(inputs)

class TruthfulStrategy(Strategy, nn.Module):
    """A strategy that plays truthful valuations."""
    def __init__(self):
        nn.Module.__init__(self)
        self.register_parameter('dummy', nn.Parameter(torch.zeros(1)))

    def forward(self, x):
        return x

    def play(self, inputs):
        return self.forward(inputs)
