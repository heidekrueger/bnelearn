class OptimizableExperiment:
    """
    Provides an optimizable function which could be used by a bayes_opt package
    
    """
    def __init__(self, experiment_type: str, hp_bounds: dict):
        """
        Format for the bounds: {'x': (2, 4), 'y': (-3, 3)}
        """
        self.experiment_type = experiment_type
        self.hp_bounds = hp_bounds

    def run(self) -> float:
        """
        will return eps_relative, for now 
        """
        return 0
