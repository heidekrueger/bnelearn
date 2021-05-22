import os
import sys
from BayesianOptimization.bayes_opt import BayesianOptimization

# put bnelearn imports after this.
# pylint: disable=wrong-import-position
sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'bnelearn'))
from bnelearn.experiment.configurations import ExperimentConfig # pylint: disable=import-error
from bnelearn.experiment.configuration_manager import ConfigurationManager  # pylint: disable=import-error

def black_box_function(x, y):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    #experiment_config, experiment_class = ConfigurationManager(experiment_type='single_item_uniform_symmetric', n_runs=2, n_epochs=3) \
    #    .get_config()
    #experiment = experiment_class(experiment_config)                                                             

    return -x ** 2 - (y - 1) ** 2 + 1


#print(black_box_function(2,1))
# Bounded region of parameter space
pbounds = {'x': (-100, 100), 'y': (-100, 100)}
pbounds = {'x': (0, 4), 'y': (-3, 3)}

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)

optimizer.maximize(init_points=5, n_iter=10)

#for i, res in enumerate(optimizer.res):
#    print("Iteration {}: \n\t{}".format(i, res))

print(optimizer.max)