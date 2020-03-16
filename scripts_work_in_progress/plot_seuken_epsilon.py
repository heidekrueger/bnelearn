import os
import sys
root_path = os.path.join(os.path.expanduser('~'), 'bnelearn/experiments')
if root_path not in sys.path:
    sys.path.append(root_path)


import numpy as np
import matplotlib.pyplot as plt
import torch

# 3D plotting
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# Params
bidder = "global"

##



logdir = os.path.join(root_path, 'LLLLGG', '6p/plots')

table = np.loadtxt(root_path + '/LLLLGG/6p/data_epsilon.txt', delimiter=',', usecols=range(3))

assert table.shape[1] == 3, 'Only case of n_items equals 2 can be plotted.'
    #$$$differentiate local and global in model and valuation

valuations = (table[:,0],table[:,1])
models_print = table[:,2]

fig = plt.figure()



ax = fig.add_subplot(1,1,1, projection='3d')
ax.plot_trisurf(
    valuations[0],
    valuations[1],
    models_print,
    color = 'yellow',
    linewidth = 0.2,
    antialiased = True
)
# ax.plot_wireframe(
#     xv[model_idx].detach().cpu().numpy(),
#     yv[model_idx].detach().cpu().numpy(),
#     models_print_wf[model_idx][:,:,input_idx].detach().cpu().numpy(),
#     rstride=4, cstride=4
# )
# Axis labeling
if bidder == "global":
    ax.set_xlim(0, 2); ax.set_ylim(0,2); ax.set_zlim(0,0.5)
else:
    ax.set_xlim(0, 1); ax.set_ylim(0,1); ax.set_zlim(0,1)
ax.set_xlabel('bundle 0 value'); ax.set_ylabel('bundle 1 value')#; ax.set_zlabel('bid')
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_title('Point epsilons for {}'.format(bidder))
ax.view_init(20, -135)

fig.tight_layout()

plt.savefig(os.path.join(logdir, 'eval_epsilon.png'))

plt.show()