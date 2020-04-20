from abc import ABC

import numpy as np
import scipy.integrate as integrate
from functools import partial
from typing import Iterable, List


from bnelearn.experiment import Experiment
from bnelearn.mechanism.auctions_combinatorial import *

from bnelearn.bidder import Bidder
from bnelearn.environment import Environment, AuctionEnvironment
from bnelearn.experiment import Experiment, GPUController
from bnelearn.experiment.configurations import ExperimentConfiguration, LearningConfiguration, LoggingConfiguration
#from bnelearn.experiment.logger import LLGAuctionLogger, LLLLGGAuctionLogger

from bnelearn.learner import ESPGLearner
from bnelearn.mechanism import FirstPriceSealedBidAuction, VickreyAuction
from bnelearn.strategy import Strategy, NeuralNetStrategy, ClosureStrategy

import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter


# TODO: Currently only implemented for uniform val
# TODO: Currently only implemented for LLG and LLLLGG
class CombinatorialExperiment(Experiment, ABC):
    payment_rule: str

    def __init__(self, n_players, n_local, n_items, experiment_config, learning_config, logging_config, gpu_config, known_bne):
        self.n_players = n_players
        self.n_local = n_local
        self.n_items = n_items

        self.payment_rule = experiment_config.payment_rule

        self.model_sharing = experiment_config.model_sharing
        if self.model_sharing:
            self.n_models = 2
            self._bidder2model: List[int] = [0] * n_local + [1] * (n_players - n_local)
        else:
            self.n_models = self.n_players
            self._bidder2model: List[int] = list(range(self.n_players))

        assert experiment_config.u_lo is not None, """Missing prior information!"""
        assert experiment_config.u_hi is not None, """Missing prior information!"""

        u_lo = experiment_config.u_lo
        if isinstance(u_lo, Iterable):
            assert len(u_lo) == n_players
            u_lo = [float(l) for l in u_lo]
        else:
            u_lo = [float(u_lo)] * n_players
        self.u_lo = u_lo

        u_hi = experiment_config.u_hi
        assert isinstance(u_hi, Iterable)
        assert len(u_hi) == n_players
        assert u_hi[1:n_local] == u_hi[:n_local-1], "local bidders should be identical"
        assert u_hi[0] < u_hi[n_local], "local bidders must be weaker than global bidder"
        self.u_hi = [float(h) for h in u_hi]

        self.plot_xmin = min(u_lo)
        self.plot_xmax = max(u_hi)
        self.plot_ymin = self.plot_xmin
        self.plot_ymax = self.plot_xmax * 1.05

        super().__init__(experiment_config,  learning_config, logging_config, gpu_config, known_bne)

    def _strat_to_bidder(self, strategy, batch_size, player_position=0):
        # TODO: this probably isn't the right place...
        # The model should know who is using it # TODO: Stefan: In my oppinion, it shouldn't...
        return Bidder.uniform(self.u_lo[player_position], self.u_hi[player_position], strategy, player_position=player_position,
                              batch_size=batch_size, n_items = self.n_items)
    # Currently only working for LLG and LLLLGG.
    def _setup_bidders(self):
        print('Setting up bidders...')
        # TODO: this seems tightly coupled with setup learners... can we change this?
        self.models = [None] * self.n_models

        for i in range(len(self.models)):
            # get id of canonical bidder for this model
            b_id = self._model2bidder[i][0]
            positive_output_point = torch.tensor([self.u_hi[b_id]]*self.n_items, dtype= torch.float)

            self.models[i] = NeuralNetStrategy(
                self.learning_config.input_length, hidden_nodes=self.learning_config.hidden_nodes,
                hidden_activations=self.learning_config.hidden_activations,
                ensure_positive_output=positive_output_point,
                output_length = self.n_items
            ).to(self.gpu_config.device)

        self.bidders = [
            self._strat_to_bidder(self.models[m_id], batch_size=self.learning_config.batch_size, player_position=i)
            for i, m_id in enumerate(self._bidder2model)]


        self.n_parameters = [sum([p.numel() for p in model.parameters()]) for model in
                             self.models]

        if self.learning_config.pretrain_iters > 0:
            print('\tpretraining...')
            #TODO: why is this on per bidder basis when everything else is on per model basis?
            for bidder in self.bidders:
                bidder.strategy.pretrain(bidder.valuations, self.learning_config.pretrain_iters)

    def _setup_learners(self):
        # TODO: the current strat_to_player kwargs is weird. Cross-check this with how values are evaluated in learner.
        # ideally, we can abstract this function away and move the functionality to the base Experiment class.
        # Implementation in SingleItem case is identical except for player position argument below.
        self.learners = []
        for m_id, model in enumerate(self.models):
            self.learners.append(ESPGLearner(model=model,
                                 environment=self.env,
                                 hyperparams=self.learning_config.learner_hyperparams,
                                 optimizer_type=self.learning_config.optimizer,
                                 optimizer_hyperparams=self.learning_config.optimizer_hyperparams,
                                 strat_to_player_kwargs={"player_position": self._model2bidder[m_id][0]}
                                 ))

    def _setup_learning_environment(self):
        self.env = AuctionEnvironment(self.mechanism,
                                      agents=self.bidders,
                                      batch_size=self.learning_config.batch_size,
                                      n_players=self.n_players,
                                      strategy_to_player_closure=self._strat_to_bidder)

    def _training_loop(self, epoch):
        # do in every iteration
        # save current params to calculate update norm
        prev_params = [torch.nn.utils.parameters_to_vector(model.parameters())
                       for model in self.models]
        # update models
        utilities = torch.tensor([
            learner.update_strategy_and_evaluate_utility()
            for learner in self.learners
        ])
        log_params = {}
        self.log_training_iteration(prev_params=prev_params, epoch=epoch,
                                           strat_to_bidder=self._strat_to_bidder,                                           
                                           utilities=utilities, 
                                           log_params=log_params)
        if epoch % 10 == 0:
            print("epoch {}, utilities: ".format(epoch))
            for i in range(len(utilities)):
                print("{}: {:.5f}".format(i, utilities[i]))
            self.log_ex_interim_regret(epoch=epoch, mechanism=self.mechanism, env=self.env, learners=self.learners, 
                                          u_lo=self.u_lo, u_hi=self.u_hi, regret_batch_size=self.regret_batch_size, regret_grid_size=self.regret_grid_size)

class LLGExperiment(CombinatorialExperiment):
    def __init__(self, experiment_config: ExperimentConfiguration, learning_config: LearningConfiguration,
                 logging_config: LoggingConfiguration, gpu_config: GPUController):

        # coupling between valuations only 0 implemented currently
        self.gamma = 0.0
        assert self.gamma == 0, "Gamma > 0 implemented yet!?"
        # Experiment specific parameters
        experiment_config.n_players = 3
        self.n_players = experiment_config.n_players # TODO: this will also be set in superclass but le'ts use it below
        
        #self.payment_rule =experiment_config.payment_rule
        # TODO: This is not exhaustive, other criteria must be fulfilled for the bne to be known! (i.e. uniformity, bounds, etc)
        known_bne = experiment_config.payment_rule in ['first_price', 'vcg', 'nearest_bid','nearest_zero', 'proxy', 'nearest_vcg']

        super().__init__(3, 2, 1, experiment_config, learning_config, logging_config, gpu_config, known_bne)


    def _setup_mechanism(self):
        self.mechanism = LLGAuction(rule = self.payment_rule)


    def _optimal_bid(self, valuation, player_position):
        if not isinstance(valuation, torch.Tensor):
            valuation = torch.tensor(valuation)

        # all core-selecting rules are strategy proof for global player:
        if self.payment_rule in ['vcg', 'proxy', 'nearest_zero', 'nearest_bid',
                                   'nearest_vcg'] and player_position == 2:
            return valuation
        # local bidders:
        if self.payment_rule == 'vcg':
            return valuation
        if self.payment_rule in ['proxy', 'nearest_zero']:
            bid_if_positive = 1 + torch.log(valuation * (1.0 - self.gamma) + self.gamma) / (1.0 - self.gamma)
            return torch.max(torch.zeros_like(valuation), bid_if_positive)
        if self.payment_rule == 'nearest_bid':
            return (np.log(2) - torch.log(2.0 - (1. - self.gamma) * valuation)) / (1. - self.gamma)
        if self.payment_rule == 'nearest_vcg':
            bid_if_positive = 2. / (2. + self.gamma) * (
                        valuation - (3. - np.sqrt(9 - (1. - self.gamma) ** 2)) / (1. - self.gamma))
            return torch.max(torch.zeros_like(valuation), bid_if_positive)
        raise ValueError('optimal bid not implemented for other rules')

    def _setup_eval_environment(self):
        bne_strategies = [
            ClosureStrategy(partial(self._optimal_bid, player_position=i))
            for i in range(self.n_players)
        ]

        bne_env = AuctionEnvironment(
            mechanism=self.mechanism,
            agents=[self._strat_to_bidder(bne_strategies[i], player_position=i, batch_size=self.logging_config.eval_batch_size)
                    for i in range(self.n_players)],
            n_players=self.n_players,
            batch_size=self.logging_config.eval_batch_size,
            strategy_to_player_closure=self._strat_to_bidder
        )

        self.bne_env = bne_env

        bne_utilities_sampled = torch.tensor(
            [bne_env.get_reward(a, draw_valuations=True) for a in bne_env.agents])

        print(('Utilities in BNE (sampled):' + '\t{:.5f}' * self.n_players + '.').format(*bne_utilities_sampled))
        print("No closed form solution for BNE utilities available in this setting. Using sampled value as baseline.")
        # TODO: possibly redraw bne-env valuations over time to eliminate bias
        self.bne_utilities = bne_utilities_sampled

    def _get_logdir(self):
        name = ['LLG', self.payment_rule]
        return os.path.join(*name)


class LLLLGGExperiment(CombinatorialExperiment):
    def __init__(self, experiment_config, learning_config: LearningConfiguration,
                 logging_config: LoggingConfiguration, gpu_config: GPUController):
        experiment_config.n_players = 6
        self.n_items = 2
        assert learning_config.input_length == 2, "Learner config has to take 2 inputs!" #TODO: what does this mean? can we move it upstream?
        
        #TODO: BNE is known for vcg
        known_bne = False

        super().__init__(6, 4, 2, experiment_config, learning_config, logging_config, gpu_config, known_bne)

    def _setup_mechanism(self):
        self.mechanism = LLLLGGAuction(rule=self.payment_rule, core_solver='NoCore', parallel=1, cuda=self.gpu_config.cuda)

    def _get_logdir(self):
        name = ['LLLLGG', self.payment_rule, str(self.n_players) + 'p']
        self.base_dir = os.path.join(*name)  # ToDo Redundant?
        return os.path.join(*name)

    def _plot(self, fig, plot_data, writer: SummaryWriter or None, epoch=None,
                xlim: list=None, ylim: list=None, labels: list=None,
                x_label="valuation", y_label="bid", fmts=['o'],
                figure_name: str='bid_function', plot_points=100):

        input_length = 2
        plot_points = self.plot_points
        lin_local = torch.linspace(self.u_lo[0], self.u_hi[0], plot_points)
        lin_global = torch.linspace(self.u_lo[4], self.u_hi[4], plot_points)
        xv = [None] * 2
        yv = [None] * 2
        xv[0], yv[0] = torch.meshgrid([lin_local, lin_local])
        xv[1], yv[1] = torch.meshgrid([lin_global, lin_global])
        valuations = torch.zeros(plot_points**2, len(self.models), input_length, device=self.gpu_config.device)
        models_print = [None] * len(self.models)
        models_print_wf = [None] * len(self.models)
        for model_idx in range(len(self.models)):
            if len(self.models) > 2:
                valuations[:,model_idx,0] = xv[0].reshape(plot_points**2)
                valuations[:,model_idx,1] = yv[0].reshape(plot_points**2)
                if model_idx>3:
                    valuations[:,model_idx,0] = xv[1].reshape(plot_points**2)
                    valuations[:,model_idx,1] = yv[1].reshape(plot_points**2)
            else:
                valuations[:,model_idx,0] = xv[model_idx].reshape(plot_points**2)
                valuations[:,model_idx,1] = yv[model_idx].reshape(plot_points**2)
            models_print[model_idx] = self.models[model_idx].play(valuations[:,model_idx,:])
            models_print_wf[model_idx] = models_print[model_idx].view(plot_points,plot_points,input_length)

        #TODO: This is a very ugly and temporary solution. Clean up! Also not even working yet!
        if figure_name == "regret":
            #plot_data[0] = plot_data[0].permute(1,0,2)
            #test = torch.tensor([t.cpu().numpy() for t in regrets]).view(len(learners), regret_batch_size, 1)
            plot_data = [plot_data[0],[plot_data[1][:,0,:],plot_data[1][:,1,:]]]
            fig, plt = self._plot_3d(plot_data, epoch, [self.plot_xmin, self.plot_xmax],
                                [self.plot_ymin, self.plot_ymax], ylim, input_length=1)   
        else: 
            fig, plt = self._plot_3d([valuations, models_print], epoch, [self.plot_xmin, self.plot_xmax],
                                [self.plot_ymin, self.plot_ymax], [self.plot_ymin, self.plot_ymax])

        self._process_figure(fig, writer, epoch, figure_name=figure_name)
        return fig

    #TODO: Fix output (currently overpallping)
    def _plot_3d(self, plot_data, epoch, xlim: list, ylim: list, zlim:list=[None,None],
                 input_length=2, x_label="valuation_0", y_label="valuation_1", z_label="bid"):
        """This implements plotting simple 2d data"""

        batch_size, n_models, n_items = plot_data[0].shape
        valuations = plot_data[0]
        bids = plot_data[1]


        plot_xmin = xlim[0]
        plot_xmax = xlim[1]
        plot_ymin = ylim[0]
        plot_ymax = ylim[1]
        plot_zmin = zlim[0]
        plot_zmax = zlim[1]

        # create the plot
        fig = plt.figure()
        for model_idx in range(n_models):
            for input_idx in range(input_length):
                ax = fig.add_subplot(n_models, input_length, model_idx*input_length+input_idx+1, projection='3d')
                ax.plot_trisurf(
                    valuations[:,model_idx,0].detach().cpu().numpy(),
                    valuations[:,model_idx,1].detach().cpu().numpy(),
                    bids[model_idx][:,input_idx].reshape(batch_size).detach().cpu().numpy(),
                    color = 'yellow',
                    linewidth = 0.2,
                    antialiased = True
                )
                # ax.plot_wireframe(
                #     xv[model_idx].detach().cpu().numpy(),
                #     yv[model_idx].detach().cpu().numpy(),
                #     models_print_wf[model_idx][:,:,input_idx].detach().cpu().numpy(),
                #     rstride=4, cstride=4
                # )
                # Axis labeling
                if n_models>2:
                    if model_idx < 4:
                        ax.set_xlim(plot_xmin, plot_xmax-(self.experiment_config.u_hi[4] - self.experiment_config.u_hi[0]))
                        ax.set_ylim(plot_ymin, plot_ymax-(self.experiment_config.u_hi[4] - self.experiment_config.u_hi[0]))
                        if plot_zmin==None:
                            ax.set_zlim(plot_zmin, self.experiment_config.u_hi[0])
                        else:
                            ax.set_zlim(plot_zmin, plot_zmax)
                    else:
                        ax.set_xlim(plot_xmin, plot_xmax)
                        ax.set_ylim(plot_ymin, plot_ymax)
                        if plot_zmin==None:
                            ax.set_zlim(plot_zmin, self.experiment_config.u_hi[4])
                        else:
                            ax.set_zlim(plot_zmin, plot_zmax)
                else:
                    if model_idx == 0:
                        ax.set_xlim(plot_xmin, plot_xmax-(self.experiment_config.u_hi[4] - self.experiment_config.u_hi[0]))
                        ax.set_ylim(plot_ymin, plot_ymax-(self.experiment_config.u_hi[4] - self.experiment_config.u_hi[0]))
                        if plot_zmin==None:
                            ax.set_zlim(plot_zmin, self.experiment_config.u_hi[0])
                        else:
                            ax.set_zlim(plot_zmin, plot_zmax)
                    else:
                        ax.set_xlim(plot_xmin, plot_xmax)
                        ax.set_ylim(plot_ymin, plot_ymax)
                        if plot_zmin==None:
                            ax.set_zlim(plot_zmin, self.experiment_config.u_hi[4])
                        else:
                            ax.set_zlim(plot_zmin, plot_zmax)

                ax.set_xlabel(x_label); ax.set_ylabel(y_label); ax.set_zlabel(z_label)
                ax.zaxis.set_major_locator(LinearLocator(10))
                ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
                ax.set_title('model {}, bundle {}'.format(model_idx, input_idx))
                ax.view_init(20, -135)
        fig.suptitle('iteration {}'.format(epoch), size=16)
        fig.tight_layout()

        return fig, plt