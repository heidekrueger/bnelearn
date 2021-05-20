class OptimizableExperiment:
    """
    Provides an optimizable function which could be used by a bayes_opt package
    
    """
    
    def __init__(self, experiment_class, experiment_config, hp_bounds: dict, n_runs_budget, save_log: bool = True):
        """
        Format for the bounds: {'x': (2, 4), 'y': (-3, 3)}
        """
        self.experiment_type = experiment_type
        self.hp_bounds = hp_bounds
        self.n_runs_budget = n_runs_budget

    def optimize(self):
        pass

    #TODO return a method with a signature containing all the HP which we would like to optimize over, creates an experiment
    # and call a _run_specific method. Will be passed to bayes_opt optimizer, when the optimize method is called.
    def _get_optimazable_method():
        pass

    def _run_specific(self) -> float:
        """
        Runs an experiment with specific set of hp's, potentially many seeds
        Will return eps_relative, for now, ideally should be able to optimize over any hp
        """
        experiment = self.experiment_class(self.experiment_config)
        cur_res = experiment.run()
        specific_metric = self._get_specific_metric(result=cur_res, metric_name='epsilon_relative')
        #save metric/result
        return specific_metric
    
    #TODO specify result type
    def _get_specific_metric(self, result, metric_name: str = 'epsilon_relative'):
        return 0


try:
    single_run_res = self.experiment.run()
    self._add_single_run_result(single_run_res)
except KeyboardInterrupt:
    print('\nKeyboardInterrupt: released memory after interruption')
    torch.cuda.empty_cache()

