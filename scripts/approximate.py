import numpy as np
import sys
import datetime
import pandas as pd
import os
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn as nn

# put bnelearn imports after this.
# pylint: disable=wrong-import-position
sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'bnelearn'))

from bnelearn.experiment.configuration_manager import ConfigurationManager
from bnelearn.strategy import NeuralNetStrategy

class BasicNN(nn.Module):
    def __init__(self, input):
        super().__init__()
        self.input_size = input

        self.stack = nn.Sequential(
            nn.Linear(input, 10),
            nn.SELU(),
            nn.Linear(10, 10),
            nn.SELU(),
            nn.Linear(10, 1),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.stack(x)
        return out


if __name__ == '__main__':

    betas = list(np.arange(0, 1, 0.2))
    values = torch.unsqueeze(torch.tensor(np.arange(0, 1, 0.001)), 1).float()
    x = torch.zeros((len(betas), len(values), 2))
    y = torch.zeros((len(betas), len(values), 1))

    log_root_dir = os.path.join(os.path.expanduser('~'), 'bnelearn', 'approximations', 'all_pay', 'regret', f'{datetime.datetime.now()}')

    os.mkdir(f"{log_root_dir}")
    os.mkdir(f"{log_root_dir}/png/")

    # check if data exists
    x_path = Path(f'{log_root_dir}/x.csv')

    if x_path.exists():
        x = pd.read_csv(f'{log_root_dir}/x.csv')
        y = pd.read_csv(f'{log_root_dir}/y.csv')

        x = x.reshape(len(betas) * len(values), 2)
        y = y.reshape(len(betas) * len(values), 1)
    else:

        # Rung single NPGA iterations
        for i, beta in enumerate(betas):

            log_dir = os.path.join(log_root_dir, f'{beta}')

            experiment_config, experiment_class = ConfigurationManager(experiment_type='single_item_symmetric_uniform_all_pay', n_runs=1, n_epochs=3500) \
                        .set_setting(n_players=2, regret=[beta, 0]) \
                        .set_learning(pretrain_iters = 500, batch_size=2**18) \
                        .set_logging(log_root_dir=log_dir, eval_batch_size=2**17, util_loss_grid_size=2**10, util_loss_batch_size=2**12, 
                                    util_loss_frequency=100000, stopping_criterion_frequency=100000, save_models=True) \
                        .set_hardware(specific_gpu=4).get_config()

            # Run experiment
            experiment = experiment_class(experiment_config)
            experiment.run()

            # Extract model
            path = list(Path(f'{log_dir}').rglob('*.pt'))[0]

                    
            model = NeuralNetStrategy(input_length=1, hidden_nodes=experiment_config.learning.hidden_nodes,
                                        hidden_activations=experiment_config.learning.hidden_activations)

            model.load_state_dict(torch.load(path))
            model.eval()

            pred = model(torch.tensor(values))
            x[i] = torch.cat((torch.ones_like(values) * beta, values), dim = 1)
            y[i] = pred

        x = x.reshape(len(betas) * len(values), 2)
        y = y.reshape(len(betas) * len(values), 1)

        # save data
        x_s = pd.DataFrame(x.numpy())
        x_s.to_csv("x.csv", index=False)
        y_s = pd.DataFrame(y.detach().numpy())
        y_s.to_csv("y.csv", index=False)

    # Generate another nn to approximate the function 
    net = BasicNN(2)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    ## Training
    for e in range(500):
        est = net(x)
        loss = loss_fn(est, y)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        print(loss)

    # Compare prediction with theoretical optimum
    beta_test = list(np.arange(0, 1, 0.2))
    optimal = [(values**2)/(2+2*beta*(1-values)) for beta in beta_test]

    net.eval()

    diffs = torch.zeros(len(beta_test))

    for i, beta in enumerate(beta_test):

        x_test = torch.cat((torch.ones_like(values) * beta, values), dim = 1)

        estimated = net(x_test)

        diffs[i] = torch.mean(optimal[i] - estimated)

        plt.plot(values, estimated.detach(), "orange")
        plt.plot(values, optimal[i], "blue")
        plt.savefig(f"{log_root_dir}/png/diffs_{beta}.png")

    print(diffs)

    

