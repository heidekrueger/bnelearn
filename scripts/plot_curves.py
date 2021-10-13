# Script for writing plotting values to disk

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from pathlib import Path
import sys

sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'bnelearn'))

from bnelearn.strategy import NeuralNetStrategy


if __name__ == '__main__':

    # Specify values
    values = torch.tensor(np.arange(0, 100, 0.1), dtype = torch.float)
    index = list(values.detach().numpy())
    values = values.reshape(len(values), 1)

    # Store estimations
    estimations = {}
    
    # Query values from models and store in df
    model_path = "/home/ewert/bnelearn/models/"

    # Determine best parameters and run NPGA to obtain models
    result_path = "/home/ewert/bnelearn/estimations/all_pay/2021-09-17 11:06:24.797734"

    results = pd.read_csv(f'{result_path}/results/results.csv')

    # Determine winning params
    print("x")

    for model_name in os.listdir(model_path):
        # Extract model path
        path = list(Path(f'{model_path}/{model_name}').rglob('*.pt'))[0]

        # Load Model
        model = NeuralNetStrategy(input_length=1, hidden_nodes=[10, 10],
                                    hidden_activations=[nn.SELU(), nn.SELU()])

        model.load_state_dict(torch.load(path))
        model.eval()

        pred = model(values)

        estimations[model_name] = pred.squeeze().tolist()

        print(f'{model_name} ... Done.')

    
    result = pd.DataFrame(estimations, index=index)

    # write to disk
    result.to_csv("./model_est.csv")



