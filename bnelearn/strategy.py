from abc import ABC, abstractmethod

import torch
import torch.nn as nn
# import torch.nn.functional as F

class Strategy(ABC):
    
    @abstractmethod
    def play(self):
        pass

class NeuralNetStrategy(Strategy, nn.Module):
    def __init__(self, input_length, depth_hidden_layer = 10):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(input_length, depth_hidden_layer)
        self.fc_out = nn.Linear(depth_hidden_layer, 1)

    def forward(self, x):
        x = self.fc1(x).tanh_()
        x = self.fc_out(x).relu_()

        return x
    
    def play(self,x):
        return self.forward(x)
    
class TruthfulStrategy(Strategy, nn.Module):
    def __init__(self, input_length):
        nn.Module.__init__(self)
        self.register_parameter('dummy',nn.Parameter(torch.zeros(1, device='cuda')))

    def forward(self, x):
        # right now specific for input length 2
        return x.matmul(torch.tensor([[1.0], [0.0]], device=x.device))
    
    def play(self, x):
        return self.forward(x)