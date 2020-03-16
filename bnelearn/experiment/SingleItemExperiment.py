from abc import ABC

from bnelearn.experiment import Experiment


# general logic and setup, plot
class SingleItemExperiment(Experiment, ABC):
    def __init__(self, name, mechanism, n_players, logging_options):
        super().__init__(name, mechanism, n_players, logging_options)


# implementation logic, e.g. model sharing. Model sharing should also override plotting function, etc.
class SymmetricPriorSingleItemExperiment(SingleItemExperiment, ABC):
    def __init__(self, name, mechanism, n_players, logging_options):
        super().__init__(name, mechanism, n_players, logging_options)


# implementation differences to symmetric case?
class AsymmetricPriorSingleItemExperiment(SingleItemExperiment, ABC):
    def __init__(self, name, mechanism, n_players, logging_options):
        super().__init__(name, mechanism, n_players, logging_options)


# known BNE
class UniformSymmetricPriorSingleItemExperiment(SymmetricPriorSingleItemExperiment):
    def __init__(self, name, mechanism, n_players, logging_options):
        super().__init__(name, mechanism, n_players, logging_options)

    def setup_players(self):
        pass

    def setup_learning_environment(self):
        pass

    def setup_learners(self):
        pass

    def setup_eval_environment(self):
        pass

    def training_loop(self, writer, e):
        pass


# known BNE + shared setup logic across runs (calculate and cache BNE
class GaussianSymmetricPriorSingleItemExperiment(SymmetricPriorSingleItemExperiment):
    def setup_players(self):
        pass

    def setup_learning_environment(self):
        pass

    def setup_learners(self):
        pass

    def setup_eval_environment(self):
        pass

    def training_loop(self, writer, e):
        pass

    def __init__(self, name, mechanism, n_players, logging_options):
        super().__init__(name, mechanism, n_players, logging_options)


# known BNE
class TwoPlayerUniformPriorSingleItemExperiment(AsymmetricPriorSingleItemExperiment):
    def setup_players(self):
        pass

    def setup_learning_environment(self):
        pass

    def setup_learners(self):
        pass

    def setup_eval_environment(self):
        pass

    def training_loop(self, writer, e):
        pass

    def __init__(self, name, mechanism, n_players, logging_options):
        super().__init__(name, mechanism, n_players, logging_options)
