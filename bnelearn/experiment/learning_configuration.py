import torch


class LearningConfiguration:
    def __init__(self,  learner_hyperparams: dict, optimizer_type: torch.optim, optimizer_hyperparams: dict,
                 input_length: int, hidden_nodes: list, hidden_activations: list, pretrain_iters: int = 500,
                 batch_size: int = 2 ** 13, eval_batch_size: int = 2 ** 12, cache_eval_actions: bool = True):
        self.learner_hyperparams = learner_hyperparams
        self.optimizer_type = optimizer_type
        self.optimizer_hyperparams = optimizer_hyperparams
        self.input_length = input_length
        self.hidden_nodes = hidden_nodes
        self.hidden_activations = hidden_activations
        self.pretrain_iters = pretrain_iters
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.cache_eval_actions = cache_eval_actions
