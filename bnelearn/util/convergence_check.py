import os
import numpy as np
import matplotlib.pyplot as plt
from bnelearn.experiment import Experiment
from bnelearn.strategy import NeuralNetStrategy
from bnelearn.util.metrics import norm_strategies

def get_distance_of_learned_strategies(experiment: Experiment, this_run_only=False, plot=True):
    """
    Calulate and return sitances between all saved models in experiment path

    TODO Nils: only supports symmetric bidders

    Args
    ----
        experiment: Experiment, for which we want to compare the saved models.
        this_run_only: bool, wheather to campare all models or just the ones from this run.
        plot: bool, save heatmap to disk.
    Returns
    -------
        distacnes, np.array
    """
    os.chdir(experiment.experiment_log_dir)

    if not this_run_only:
        os.chdir('..')

    print(os.path.abspath(os.curdir))
    valuations = experiment.bidders[0].draw_valuations_()

    model_paths = []
    for root, _, files in os.walk(os.curdir):
        for name in files:
            if name.find('model') != -1 and name.find('.pt') != -1:
                model_paths.append(os.path.join(root, name))
    n_models = len(model_paths)

    models = [NeuralNetStrategy.load(p).to(experiment.gpu_config.device) for p in model_paths]
    distances = - np.ones((n_models, n_models))
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            distances[i, j] = norm_strategies(m1, m2, valuations)

    model_paths = [p[2:p[2:].find('/')+2] for p in model_paths]

    # plot
    fig, ax = plt.subplots()
    ax.imshow(distances)
    ax.set_xticks(np.arange(n_models))
    ax.set_yticks(np.arange(n_models))
    ax.set_xticklabels(model_paths)
    ax.set_yticklabels(model_paths)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(model_paths)):
        for j in range(len(model_paths)):
            _ = ax.text(j, i, round(distances[i, j], 4), ha="center", va="center", color="w")
    ax.set_title("Distances")
    fig.tight_layout()
    plt.savefig('distances.png')

    return distances
