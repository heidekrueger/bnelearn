from abc import ABC

from bnelearn.experiment import Experiment


class MultiUnitExperiment(Experiment, ABC):
    def __init__(self, name, mechanism, n_players, logging_options):
        super().__init__(name, mechanism, n_players, logging_options)


# exp_no==0
class MultiItemVickreyAuction(MultiUnitExperiment):
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


# exp_no==1, BNE continua
class MultiItemUniformPriceAuction2x2(MultiUnitExperiment):
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


# exp_no==2
class MultiItemUniformPriceAuction2x3limit2(MultiUnitExperiment):
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


# exp_no==4
class MultiItemDiscriminatoryAuction2x2(MultiUnitExperiment):
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


# exp_no==5
class MultiItemDiscriminatoryAuction2x2CMV(MultiUnitExperiment):
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


# exp_no==6, two BNE types, BNE continua
class FPSBSplitAwardAuction2x2(MultiUnitExperiment):
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
