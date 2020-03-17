from abc import ABC

from bnelearn.experiment import Experiment


class CombinatorialExperiment(Experiment, ABC):
    def __init__(self, name, mechanism_type, n_players, logging_options):
        super().__init__(name, mechanism_type, n_players, logging_options)


# mechanism/bidding implementation, plot, bnes
class LLGExperiment(CombinatorialExperiment):
    def __init__(self, name, mechanism_type, n_players, logging_options):
        super().__init__(name, mechanism_type, n_players, logging_options)

    def setup_experiment_domain(self):
        pass

    def setup_learning_environment(self):
        pass

    def setup_learners(self):
        pass

    def setup_eval_environment(self):
        pass

    def training_loop(self, writer, e):
        pass


# mechanism/bidding implementation, plot
class LLLLGGExperiment(CombinatorialExperiment):
    def __init__(self, name, mechanism_type, n_players, logging_options):
        super().__init__(name, mechanism_type, n_players, logging_options)

    def setup_experiment_domain(self):
        pass

    def setup_learning_environment(self):
        pass

    def setup_learners(self):
        pass

    def setup_eval_environment(self):
        pass

    def training_loop(self, writer, e):
        pass
