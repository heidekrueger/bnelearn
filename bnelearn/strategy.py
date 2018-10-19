from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.nn.functional as F

class Strategy(ABC):
    
    @abstractmethod
    def play(self):
        pass

class NeuralNetStrategy(Strategy, nn.Module):
    def __init__(self, input_length, depth_hidden_layer = 10):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(input_length, depth_hidden_layer)
        self.fc2 = nn.Linear(depth_hidden_layer, depth_hidden_layer)
        self.fc_out = nn.Linear(depth_hidden_layer, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=.1)
        

    def forward(self, x):
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc2(x))
        x = self.fc_out(x).relu()

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
        print(x.shape)

        return x
