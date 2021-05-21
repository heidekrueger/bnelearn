import os
import sys
import numpy as np
from bayes_opt import BayesianOptimization


# put bnelearn imports after this.
# pylint: disable=wrong-import-position
sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'bnelearn'))

from bnelearn.experiment.configurations import ExperimentConfig # pylint: disable=import-error
from bnelearn.experiment.configuration_manager import ConfigurationManager  # pylint: disable=import-error

class OptimizableExperiment:
    """
    Optimize hp's using bayes_opt    
    """
    
    def __init__(self, experiment_class, experiment_config: ExperimentConfig, hp_bounds: dict, n_runs_budget = 5, save_log: bool = True):
        """
        Format for the bounds: {'x': (2, 4), 'y': (-3, 3)}
        """
        self.experiment_class = experiment_class
        self.experiment_config = experiment_config        
        self.hp_bounds = hp_bounds
        self.n_runs_budget = n_runs_budget
    
    # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    def optimize(self):
        optimizer = BayesianOptimization(f=self._optimizable_function, pbounds=self.hp_bounds, verbose=2, random_state=1)
        optimizer.maximize(init_points=2, n_iter=3,)

        print(optimizer.max)

        for i, res in enumerate(optimizer.res):
            print("Iteration {}: \n\t{}".format(i, res))

    def _optimizable_function(self, lr=1e-3) -> float:
        self.experiment_config.learning.optimizer_hyperparams['lr'] = lr
        
        experiment = self.experiment_class(self.experiment_config)
        cur_res = experiment.run()
        specific_metric = self._get_specific_metric(result=cur_res, metric_name='epsilon_relative')
        #save metric/result
        return specific_metric
    
    #TODO specify result type
    def _get_specific_metric(self, result, metric_name: str = 'epsilon_relative'):
        eps_rel = []
        for metric_entry in result:
            if metric_entry[2] == 'eval/epsilon_relative':                    
                print(metric_entry)
                eps_rel.append(metric_entry[4])                    

            np.array(eps_rel)
            print(np.average(eps_rel))

        return 0

try:
    log_root_dir = os.path.join(os.path.expanduser('~'), 'bnelearn', 'experiments')

    #n_runs here means # of seeds
    experiment_config, experiment_class = ConfigurationManager(experiment_type='single_item_uniform_symmetric', n_runs=2,
                                                                n_epochs=3) \
            .set_setting(risk=1.1)\
            .set_logging(log_root_dir=log_root_dir, save_tb_events_to_csv_detailed=True)\
            .set_learning(pretrain_iters=5) \
            .set_logging(eval_batch_size=2**22).set_hardware(specific_gpu=7).get_config()

    experiment = OptimizableExperiment(experiment_class=experiment_class, experiment_config=experiment_config, hp_bounds={}, n_runs_budget=20)
    experiment._optimizable_function()

except KeyboardInterrupt:
    print('\nKeyboardInterrupt: released memory after interruption')
    torch.cuda.empty_cache()

