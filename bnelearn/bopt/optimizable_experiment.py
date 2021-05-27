import os
import sys
import numpy as np
from BayesianOptimization.bayes_opt import BayesianOptimization

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


class OptimizableExperiment:
    """
    Optimize hp's using bayes_opt    
    """

    def __init__(
        self,
        experiment_class,
        experiment_config: ExperimentConfig,
        hp_bounds: dict,
        n_runs_budget=5,
        save_log: bool = True,
        metric_name: str = "epsilon_relative",
    ):
        """
        Format for the hp_bounds: {'x': (2, 4), 'y': (-3, 3)}, where x and y are some hp's
        n_runs_budget is the number of effective hp configurations to check, to get the full number of runs, 
        multiply by the number of seeds per run
        """
        print(type(experiment_class))
        self.experiment_class = experiment_class
        self.experiment_config = experiment_config
        self.hp_bounds = hp_bounds
        self.n_runs_budget = n_runs_budget
        self.metric_name = metric_name

    # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    def optimize(self, init_points: int = 3, minimize: bool = True):
        optimizer = BayesianOptimization(
            f=self._get_optimizable_function(minimize=minimize),
            pbounds=self.hp_bounds,
            verbose=2,
            random_state=1,
        )
        optimizer.maximize(init_points=init_points, n_iter=self.n_runs_budget)

        for i, res in enumerate(optimizer.res):
            print("Iteration {}: \n\t{}".format(i, res))

        print(optimizer.max)

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
            self.experiment_config.learning.hidden_activations = [activation_function] * hidden_layers
            self.experiment_config.learning.optimizer_hyperparams["lr"] = lr
            self.experiment_config.learning.learner_hyperparams['population_size'] = population_size
            self.experiment_config.learning.learner_hyperparams['sigma'] = sigma

            
            experiment = self.experiment_class(self.experiment_config)
            res = experiment.run()
            res = res.to_numpy()
            specific_metric = self._get_specific_metric(result=res)
            # save metric/result
            if minimize == True:
                return -specific_metric
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
                print(metric_entry)

        np.array(eps_rel)
        averaged_metric = np.average(eps_rel)
        print("Averaged:" + str(averaged_metric))

        return averaged_metric


try:
    log_root_dir = os.path.join(os.path.expanduser("~"), "bnelearn", "experiments")

    # n_runs here means # of seeds
    experiment_config, experiment_class = (
        ConfigurationManager(
            experiment_type="single_item_uniform_symmetric", n_runs=3, n_epochs=100
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
        n_runs_budget=10,
    )
    experiment.optimize()

except KeyboardInterrupt:
    print("\nKeyboardInterrupt: released memory after interruption")
    torch.cuda.empty_cache()

