import os
import sys
import numpy as np
import torch

# put bnelearn imports after this.
# pylint: disable=wrong-import-position
sys.path.append(os.path.realpath("."))
sys.path.append(os.path.join(os.path.expanduser("~"), "bnelearn"))
from bnelearn.experiment.configurations import (
    ExperimentConfig,
)  # pylint: disable=import-error
from bnelearn.experiment.configuration_manager import (
    ConfigurationManager,
)  # pylint: disable=import-error

from BayesianOptimization.bayes_opt import BayesianOptimization
from BayesianOptimization.bayes_opt.logger import JSONLogger
from BayesianOptimization.bayes_opt.event import Events


class OptimizableExperiment:
    """
    Optimize hyperparameters of the experiment using bayes_opt    
    """

    def __init__(
        self,
        experiment_class,
        experiment_config: ExperimentConfig,
        minimize: bool,  # should we min or max the given metric?
        hp_bounds: dict,
        metric_name: str = "epsilon_relative",
        n_runs_budget=5,
        verbose: bool = True,
        log: bool = True,
    ):
        """
        Format for the hp_bounds: {'x': (2, 4), 'y': (-3, 3)}, where x and y are some hp's
        n_runs_budget is the number of effective hp configurations to check, to get the full number of runs, 
        multiply by the number of seeds per run
        """
        self.experiment_class = experiment_class
        self.experiment_config = experiment_config
        self.hp_bounds = hp_bounds
        self.n_runs_budget = n_runs_budget
        self.metric_name = metric_name
        self.verbose = verbose

        self.optimizer = BayesianOptimization(
            f=self._get_optimizable_function(minimize=minimize),
            pbounds=self.hp_bounds,
            verbose=2
            if verbose
            else 0,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=1,
        )

        if log:
            logger = JSONLogger(path="./logs.json")
            self.optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    def optimize(self, init_points: int = 3):
        self.optimizer.maximize(init_points=init_points, n_iter=self.n_runs_budget)

        if self.verbose:
            for i, res in enumerate(self.optimizer.res):
                print("Iteration {}: \n\t{}".format(i, res))
            print(self.optimizer.max)

    # maybe population size should be described as 2 ** k, where k varies from 1 to 7
    def _get_optimizable_function(self, minimize: bool):
        # only those hp's which are specified in the hp_bounds would be passed, parameter names should match exactly
        def optimizable_function(
            lr: float = 1e-3,  # learning rate
            population_size: int = 64,  # from 2 to 128 is reasonable (>128 will run really long), this is the # of perturbations in the ES
            sigma: float = 1.0,  # from 0.1 to 10 should be reasonable, sigma is the magnitude of the perturbations in the ES
            hidden_nodes: int = 10,
            hidden_layers: int = 2,
        ) -> float:
            # The discrete params has to be rounded since GP will treat them as continuous and pass floats
            population_size = round(population_size)
            hidden_nodes = round(hidden_nodes)
            hidden_layers = round(hidden_layers)
            hidden_nodes = [
                hidden_nodes
            ] * hidden_layers  # that's the real parameter format we use
            activation_function = self.experiment_config.learning.hidden_activations[0]

            # Just a hack, basically, but do we really want to handle different actiavations for each layer?
            self.experiment_config.learning.hidden_activations = [g
                activation_function
            ] * hidden_layers
            self.experiment_config.learning.optimizer_hyperparams["lr"] = lr
            self.experiment_config.learning.learner_hyperparams[
                "population_size"
            ] = population_size
            self.experiment_config.learning.learner_hyperparams["sigma"] = sigma

            experiment = self.experiment_class(self.experiment_config)
            res = experiment.run()            
            specific_metric = self._get_specific_metric(result=res.to_numpy())

            torch.cuda.empty_cache()            
            # save metric/result
            if minimize == True:
                return -specific_metric
            else:
                return specific_metric

        return optimizable_function

    def set_new_bounds(self, new_bounds: dict):
        """
        During the optimization process you may realize the bounds chosen for some parameters are not adequate. For these situations you can 
        invoke the method to alter them. You can pass any combination of existing parameters and their associated new bounds.
        """
        self.optimizer.set_bounds(new_bounds=new_bounds)

    def probe_point(self, params: dict, lazy: bool = True):
        """
        Checks a specific point (hyperparameter set)
        (lazy=True), means these point will be evaluated only the next time you call maximize. (immediately otherwise)
        This probing process happens before the gaussian process takes over.
        """
        self.optimizer.probe(
            params=params, lazy=lazy,
        )

    def _get_specific_metric(self, result: np.ndarray) -> float:
        """
        Takes experiment result and returns an avereged across all seeds specified metric.
        Possible metric names include: utilities, update_norm, utility_vs_bne, epsilon_relative, epsilon_absolute,
        L_2, L_inf, util_loss_ex_ante, util_loss_ex_interim, estimated_relative_ex_ante_util_loss
        """
        eps_rel = []
        for metric_entry in result:
            if metric_entry[2] == "eval/{}".format(self.metric_name):
                eps_rel.append(metric_entry[4])
                print(metric_entry)

        np.array(eps_rel)
        averaged_metric = np.average(eps_rel)
        print("Averaged:" + str(averaged_metric))

        return averaged_metric


try:
    #the full logs for the experiments, not the bayes_opt log
    log_root_dir = os.path.join(os.path.expanduser("~"), "bnelearn", "experiments")

    # n_runs here means # of seeds
    experiment_config, experiment_class = (
        ConfigurationManager(
            experiment_type="single_item_uniform_symmetric", n_runs=1, n_epochs=10
        )
        .set_setting(risk=1.1)
        .set_logging(log_root_dir=log_root_dir, save_tb_events_to_csv_detailed=True)
        .set_learning(pretrain_iters=5)
        .set_logging(eval_batch_size=2 ** 22)
        .set_hardware(specific_gpu=7)
        .get_config()
    )

    experiment = OptimizableExperiment(
        experiment_class=experiment_class,
        experiment_config=experiment_config,
        hp_bounds={"lr": (0.0001, 0.01)},
        metric_name = "epsilon_relative",
        n_runs_budget=10,
        minimize=True,
        verbose=True,
        log=True
    )
    experiment.optimize(init_points=3)

except KeyboardInterrupt:
    print("\nKeyboardInterrupt: released memory after interruption")
    torch.cuda.empty_cache()
