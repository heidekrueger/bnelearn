from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

## false positive on torch.tensor()
#pylint: disable=E1102

class Strategy(ABC):
    
    @abstractmethod
    def play(self):
        pass

# TODO (backing up first)
#class NeuralNetStrategy(Strategy, nn.Module):
    # abstract class for a Strategy that is also an nn module.
    # useful for abstracting away shared initialization stuff

#    def __init__(self):
#        nn.Module.__init__(self)

#    def _disable_gradients():


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
        x = F.tanh(self.fc1(x))
        #x = F.tanh(self.fc2(x))
        #x = self.tanh(self.fc1(x))
        x = self.fc_out(x)

        return x
    
    def play(self,x):
        return self.forward(x)


class TruthfulStrategy(Strategy, nn.Module):
    def __init__(self, input_length):
        nn.Module.__init__(self)
        self.register_parameter('dummy',nn.Parameter(torch.zeros(1)))

    def forward(self, x):
        # simply play first input
        # right now specific for input length 2!
        return x.matmul(torch.tensor([[1.0], [0.0]], device=x.device))
    
    def play(self, x):
        return self.forward(x)

class FpsbBneStrategy(Strategy, nn.Module):
    def __init__(self, input_length):
        nn.Module.__init__(self)
        self.register_parameter('dummy', nn.Parameter(torch.zeros(1)))
    
    def forward(self, x):
        # assumes valuation in first input, n_players in second
        raise NotImplementedError()

class RandomStrategy(Strategy, nn.Module):
    def __init__(self, input_length, lo=0, hi=10):
        nn.Module.__init__(self)
        self.register_parameter('dummy', nn.Parameter(torch.zeros(1)))
    
    def forward(self, x):        
        return x
