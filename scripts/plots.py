import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import sys

sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'bnelearn'))

from bnelearn.strategy import NeuralNetStrategy 

#import NeuralNetStrategy from Projekte.bnelearn.bnelearn.strategy

colors = {0:(0/255.,150/255.,196/255.),
          1:(248/255.,118/255.,109/255.),
          2:(255/255.,215/255.,130/255.),
          3:(150/255.,120/255.,170/255.)}

save = True

# ------------------------ CROWDSOURCING CONTEST ----------------------------- #
# TODO - define obs, bids, bne
"""
x = np.linspace(0,1,100)
obs =
bids =
bne =
"""

# load model
paths = ["/home/ewert/bnelearn/abstract_plots/crowdsourcing/3/1.0/2022-04-14 Thu 10.06/01 10:42:03 1/models/model_0.pt",
"/home/ewert/bnelearn/abstract_plots/crowdsourcing/3/0.800000011920929/2022-04-14 Thu 10.07/04 12:24:43 4/models/model_0.pt",
"/home/ewert/bnelearn/abstract_plots/crowdsourcing/3/0.6000000238418579/2022-04-14 Thu 10.08/02 11:12:00 2/models/model_0.pt"]

obs = torch.linspace(0,1,100).unsqueeze(-1)
bids = []

for i, p in enumerate(paths):

    model = NeuralNetStrategy(1, [10, 10], [nn.SELU(), nn.SELU()])

    model.load_state_dict(torch.load(p))    
    model.eval()
    model.cpu()





    # predict
    bids.append(model(obs))  
    torch.cuda.empty_cache()   
    del model       


# define bne
def c(valuation: torch.Tensor, v1: float = 1, v2 = 0, N: int = 0, player_position=0, **kwargs):
    #return torch.relu(0.63 * (valuation ** 2 - 2/3 * valuation ** 3) - 0.37 * valuation ** 2)
    #return torch.relu(v1 * 2 * ((valuation ** 2)/2 - (valuation ** 3)/3) + v2 * 2 * ((valuation ** 2)/2 - (2*valuation**3)/3))

    # old for n = 3 and general m
    # a = lambda m, v: 2/(1-m) * ((m**3)/(6*(1-m)) + (v ** 3)/(3*(1-m)) - (m*v**2)/(2*(1-m)))
    # b = lambda m, v: 2/(1-m) * (-(m**3)/(3*(1-m)) - (m ** 2)/2 - (2 * v **3)/(3*(1-m)) + (m * v ** 2)/(1-m) + (v ** 2)/2)

    # return v1 * a(m, valuation) + v2 * b(m, valuation)
    a = lambda v, N: (N-1)/N * v ** N
    b = lambda v, N: (N-1) * (((N-2) * v ** (N-1))/(N-1) + (v**N)/N - v**N)
    return torch.relu(v1 * a(valuation, N) + v2 * b(valuation, N))

bne = [c(obs, x, 1-x, 0) for x in [1.0, 0.8, 0.6]]

labels = {0: '$ w_1 = 1.0$',
          1: '$w_1 = 0.8$',
          2: '$w_1 = 0.6$'}


fig = plt.figure(figsize=(5.5, 5.5))
ax = fig.add_subplot(111)

# BNE
for k in range(3):
    plt.plot(obs, bne[k], color='k', linestyle='--', linewidth=1.5, alpha=.7)
# BNE Label
plt.plot([], [], color='k', linestyle='--', linewidth=1.5, label = 'BNE' )


# STRATEGIES
for k in range(3):
    # observation and bids from strategies
    plt.scatter(obs[k], bids[k], s = 5, color = colors[k])
    # labels
    plt.scatter([], [], s = 25, color = colors[k], label=labels[k])


ax.set_aspect(1)
plt.ylabel('effort $e$', fontsize=14)
plt.xlabel('valuation (type) $v$',fontsize=14)
plt.title('NPGA - Crowdsourcing',fontsize=16)

plt.xlim(-0.05,1.05)
plt.ylim(-0.05, 0.75)

plt.legend(fontsize=13)
plt.grid(alpha=.3)

if save:
    plt.savefig('crowdsourcing_3.pdf')

plt.show()

print(2)

# --------------------------- TULLOCK CONTEST -------------------------------- #

# TODO - define obs, bids, bne
"""
obs =
bids =
"""

labels = {0: '$ \epsilon = 0.5$',
          1: '$ \epsilon = 1$',
          2: '$ \epsilon = 2$',
          3: '$ \epsilon = 5$',
         }

fig = plt.figure(figsize=(5.5, 5.5))
ax = fig.add_subplot(111)

# STRATEGIES
for k in np.arange(4):
    # observation and bids from strategies
    plt.scatter(obs[k], bids[k], s = 5, color = colors[k])
    # labels
    plt.scatter([], [], s = 25, color = colors[k], label=labels[k])



ax.set_aspect(1.6)
plt.ylabel('effort $e$', fontsize=14)
plt.xlabel('valuation (type) $v$',fontsize=14)
plt.title('SODA',fontsize=16)

plt.xlim(-0.05, 1.05)
plt.ylim(-0.03, 0.45)

plt.legend(fontsize=13)
plt.grid(alpha=.3)

if save:
    plt.savefig('tullock_contest_2.pdf')

plt.show()
