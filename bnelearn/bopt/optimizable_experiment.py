import os
import sys
import numpy as np
import torch
import json

# pylint: disable=wrong-import-position
sys.path.append(os.path.realpath("."))
sys.path.append(os.path.join(os.path.expanduser("~"), "bnelearn"))

# pylint: disable=import-error
from bnelearn.experiment.configurations import ExperimentConfig

from BayesianOptimization.bayes_opt.observer import _Tracker
from bayes_opt.util import load_logs

from BayesianOptimization.bayes_opt import (
    BayesianOptimization,
    JSONLogger,
    Events,
    ScreenLogger,
)

# from BayesianOptimization.bayes_opt.logger import JSONLogger
# from BayesianOptimization.bayes_opt.event import Events


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
        seed: int = 1,
    ):
        """
        Format for the hp_bounds: {'x': (2, 4), 'y': (-3, 3)}, where x and y are some hp's
        n_runs_budget is the number of effective hp configurations to check, to get the full number of runs, 
        multiply by the number of seeds per run

        Seed: Determines random number generation used to initialize the centers.
        Pass an int for reproducible results across multiple function calls.
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
            random_state=seed,
        )

        # TODO make flexible, check not to override old logs, test with runing and restarting
        if log:
            logger = self.Observer(path="bnelearn/bopt/logs/logs.json")
            self.optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
            # self.optimizer.subscribe(Events.OPTIMIZATION_START, logger)
            # self.optimizer.subscribe(Events.OPTIMIZATION_END, logger)

    def optimize(self, init_points: int = 3):
        self.optimizer.maximize(init_points=init_points, n_iter=self.n_runs_budget)

        if self.verbose:
            for i, res in enumerate(self.optimizer.res):
                print("Iteration {}: \n\t{}".format(i, res))
            print(self.optimizer.max)

        return self.optimizer.max

    def load_from_logs(self, path: str):
        load_logs(self.optimizer, logs=[path])  # TODO test

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

    # maybe population size should be described as 2 ** k, where k varies from 1 to 7
    def _get_optimizable_function(self, minimize: bool):
        # only those hp's which are specified in the hp_bounds would be passed, parameter names should match exactly
        # TODO extend for all the possible hp's
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
            self.experiment_config.learning.hidden_activations = [
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
            if minimize:
                return -1 * specific_metric
            else:
                return specific_metric

        return optimizable_function

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
                # print(metric_entry)

        np.array(eps_rel)
        averaged_metric = np.average(eps_rel)
        print("Averaged metric: " + str(averaged_metric))

        return averaged_metric

    class Observer(_Tracker):
        """
        """

        def __init__(self, path):
            self.path = path
            super(type(self), self).__init__()

        def update(self, event, instance):
            self.event = event
            self.instance = instance
            if event == Events.OPTIMIZATION_START:
                self._on_opt_start()
            if event == Events.OPTIMIZATION_STEP:
                self._on_opt_step()
            if event == Events.OPTIMIZATION_END:
                self._on_opt_end()

        def _on_opt_start(self):
            pass

        def _on_opt_step(self):
            print("_on_opt_step")
            data = dict(self.instance.res[-1])
            now, time_elapsed, time_delta = self._time_metrics()
            data["datetime"] = {
                "datetime": now,
                "elapsed": time_elapsed,
                "delta": time_delta,
            }

            with open(self.path, "a+") as f:
                f.write(json.dumps(data) + "\n")

            self._update_tracker(self.event, self.instance)

        def _on_opt_end(self):
            pass
