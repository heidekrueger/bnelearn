"""
Method for getting a feel of mechanism computing times...
"""
import matplotlib.pyplot as plt
import torch
import numpy as np
import sys, os
sys.path.append(os.path.realpath('.'))
from bnelearn.mechanism import MultiItemDiscriminatoryAuction, \
    MultiItemUniformPriceAuction, MultiItemVickreyAuction
from time import time



batch_size = 2**15
n_agents_list = [i for i in range(2, 41)]
n_items_list = [i for i in range(1, 42)]

specific_gpu = 7
if specific_gpu:
    torch.cuda.set_device(specific_gpu)
mechanism = MultiItemVickreyAuction(cuda=True)

def get_mechanism_performance(mechanism, batch_size, n_agents, n_items):
    #prepare simulated bids
    bids = torch.rand(batch_size, n_agents, n_items, device='cuda')
    bids_flat = bids.reshape(batch_size, n_agents*n_items)
    sorted_bids = torch.sort(bids_flat, descending=True)[0]
    bids = sorted_bids.reshape_as(bids)

    # run and measure mechanism
    start_time = time()
    mechanism.run(bids)
    return time() - start_time

torch.cuda.reset_max_memory_allocated(device='cuda')
timings = np.zeros((len(n_agents_list), len(n_items_list)))
for i_agent, n_agents in enumerate(n_agents_list):
    for i_item, n_items in enumerate(n_items_list):
        timings[i_agent,i_item] = get_mechanism_performance(mechanism, batch_size, n_agents, n_items)

# plotting
auction_type_str = str(type(mechanism))
auction_type_str = str(auction_type_str[len(auction_type_str) \
                       - auction_type_str[::-1].find('.'):-2])
plt.imshow(timings, cmap='plasma', vmin=0.0, vmax=0.11,
           interpolation='nearest')
plt.title(auction_type_str)
plt.ylabel('$n_{agents}$')
plt.xlabel('$n_{items}$')
plt.yticks(range(0,len(n_agents_list), 5), n_agents_list[::5])
plt.xticks(range(0,len(n_items_list), 5), n_items_list[::5])
cb = plt.colorbar()
cb.set_label('runtime in secs (for batch size of {})'.format(batch_size))
plt.show()

print('max_memory_allocated = {}mb'.format(torch.cuda.max_memory_allocated(device='cuda') / 1e+6))
