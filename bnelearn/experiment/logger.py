import os
import time
from sys import platform
from time import perf_counter as timer

from bnelearn.experiment.configurations import * # needed for e.g json serialization 
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import bnelearn.util.metrics as metrics

from typing import Iterable, List, Callable
from bnelearn.experiment.configurations import ExperimentConfig
from bnelearn.util.custom_summary_writer import CustomSummaryWriter
from bnelearn.environment import Environment
from bnelearn.bidder import Bidder

class Logger:
    """
    Helper class encapsulating all the logging and plotting logic for the Experiment
    """
    def __init__(self, config: ExperimentConfig, known_bne: bool, plot_bounds: dict,  _model2bidder, n_models,
                model_names, logdir_hierarchy, sampler, plotter, optimal_bid, evaluation_env = None, valuation_size=None, epoch_logger=None):
        self.config = config
        self.logging = config.logging
        self.learning = config.learning
        
        # fields depending on initialization in subclasses
        self.bne_env = evaluation_env
        self.valuation_size = valuation_size
        self.env = None
        self.bne_utilities = None
        self._model2bidder = _model2bidder
        self.n_models = n_models   
        self.model_names = model_names
        self.models = None
        self.sampler = sampler
        self._optimal_bid = optimal_bid
        self.plot_xmin = plot_bounds['plot_xmin']
        self.plot_xmax = plot_bounds['plot_xmax']
        self.plot_ymin = plot_bounds['plot_ymin']
        self.plot_ymax = plot_bounds['plot_ymax']        
        self.known_bne = known_bne
        if not known_bne:
            self.logging.log_metrics['opt'] = False

        # This is the best I can think of without recreating logger subclasses just because there is one time (in combinatorial auctions),
        # when there is additional logic in logging epoch info     
        self.epoch_logger = epoch_logger

        # A method which would get an approptiate implementation from an experiment subcalass
        self._plot = plotter 

        # Global Stuff that should be initiated here
        self.plot_frequency = self.logging.plot_frequency
        self.plot_points = min(self.logging.plot_points, self.learning.batch_size)

        # Everything that will be set up per run initiated with none
        #self.run_log_dir = None
        self.writer = None
        self.overhead = 0.0

         # These are set on first _log_experiment
        self.v_opt: torch.Tensor = None
        self.b_opt: torch.Tensor = None

        self._hparams_metrics = {}
        ### Save locally - can haves
        # Logging
        if self.logging.util_loss_batch_size is not None:
            self.util_loss_batch_size = self.logging.util_loss_batch_size
        if self.logging.util_loss_grid_size is not None:
            self.util_loss_grid_size = self.logging.util_loss_grid_size
        self.n_parameters = None
        self._cur_epoch_log_params = {}
        
        self.run_log_dir = None        
        # sets log dir for experiment. Individual runs will log to subdirectories of this.
        self.experiment_log_dir = os.path.join(self.logging.log_root_dir,
                                               logdir_hierarchy,
                                               self.logging.experiment_dir)
    


    def init_new_run(self, models, env, bne_utilities):
        self.models = models
        self.env = env
        self.bne_utilities = bne_utilities

        if self.logging.log_metrics['opt'] and hasattr(self, 'bne_env'):
            if not isinstance(self.bne_env, list):
                # TODO Nils: should perhaps always be a list, even when there is only one BNE
                # TODO Stefan: Yes, we should not do any type conversions here, these should be lists from the beginning.
                self.bne_env: List[Environment] = [self.bne_env]
                self._optimal_bid: List[Callable] = [self._optimal_bid]
                self.bne_utilities = [self.bne_utilities]

            self._setup_plot_equilibirum_data()

        is_ipython = 'inline' in plt.get_backend()
        if is_ipython:
            from IPython import display  # pylint: disable=unused-import,import-outside-toplevel
        plt.rcParams['figure.figsize'] = [8, 5]

        print('Stating run...')
    
        # Create summary writer object and create output dirs if necessary
        output_dir = self.run_log_dir
        os.makedirs(output_dir, exist_ok=False)
        if self.logging.save_figure_to_disk_png:
            os.mkdir(os.path.join(output_dir, 'png'))
        if self.logging.save_figure_to_disk_svg:
            os.mkdir(os.path.join(output_dir, 'svg'))
        if self.logging.save_models:
            os.mkdir(os.path.join(output_dir, 'models'))
        self.writer = CustomSummaryWriter(output_dir, flush_secs=30)

        print('\tLogging to {}.'.format(output_dir))

        self.fig = plt.figure()

        tic = timer()
        # self._log_experiment_params()
        # self._log_hyperparams()
        self.save_experiment_config(self.experiment_log_dir, self.config)
        self.log_git_commit_hash()
        elapsed = timer() - tic
        
        self.overhead += elapsed


    def set_current_run(self, run_id: str, seed):
        t = time.strftime('%T ')
        if platform == 'win32':
            t = t.replace(':', '.')

        self.run_log_dir = os.path.join(self.experiment_log_dir, f'{run_id:02d} ' + t + str(seed))


    def log_epoch(self, epoch, utilities, prev_params):
        """
        Checks which metrics have to be logged and performs logging and plotting.
        Returns:
            - elapsed time in seconds
            - Stefan todos / understanding quesitons
            - TODO: takes log_params. can it be
        """
        if self.epoch_logger:
            self.epoch_logger(writer=self.writer, env=self.env, valuation_size=self.valuation_size, epoch=epoch)

        # pylint: disable=attribute-defined-outside-init
        self._cur_epoch_log_params = {
            'utilities': utilities.detach(),
            'prev_params': prev_params
        }

        start_time = timer()

        # calculate infinity-norm of update step
        new_params = [torch.nn.utils.parameters_to_vector(model.parameters()) for model in self.models]
        self._cur_epoch_log_params['update_norm'] = [
            (new_params[i] - self._cur_epoch_log_params['prev_params'][i]).norm(float('inf'))
            for i in range(self.n_models)]
        del self._cur_epoch_log_params['prev_params']

        # logging metrics
        # TODO: should just check if logging is enabled in general... if bne_exists and we log, we always want this
        if self.known_bne and self.logging.log_metrics['opt']:
            utility_vs_bne, epsilon_relative, epsilon_absolute = self._calculate_metrics_known_bne()
            L_2, L_inf = self._calculate_metrics_action_space_norms()
            for i in range(len(self.bne_env)):
                n = '_bne' + str(i + 1) if len(self.bne_env) > 1 else ''
                self._cur_epoch_log_params['utility_vs_bne' + (n if n == '' else n[4:])] \
                    = utility_vs_bne[i]
                self._cur_epoch_log_params['epsilon_relative' + n] = epsilon_relative[i]
                self._cur_epoch_log_params['epsilon_absolute' + n] = epsilon_absolute[i]
                self._cur_epoch_log_params['L_2' + n] = L_2[i]
                self._cur_epoch_log_params['L_inf' + n] = L_inf[i]

        if self.logging.log_metrics['util_loss'] and (epoch % self.logging.util_loss_frequency) == 0:
            create_plot_output = epoch % self.logging.plot_frequency == 0
            self._cur_epoch_log_params['util_loss_ex_ante'], \
            self._cur_epoch_log_params['util_loss_ex_interim'], \
            self._cur_epoch_log_params['estimated_relative_ex_ante_util_loss'] = \
                self._calculate_metrics_util_loss(create_plot_output, epoch)
            print("\tcurrent est. ex-interim loss:" + str(
                [f"{l.item():.4f}" for l in self._cur_epoch_log_params['util_loss_ex_interim']]))

        if self.logging.log_metrics['efficiency'] and (epoch % self.logging.util_loss_frequency) == 0:
            self._cur_epoch_log_params['efficiency'] = \
                self.env.get_efficiency(self.env)

        if self.logging.log_metrics['revenue'] and (epoch % self.logging.util_loss_frequency) == 0:
            self._cur_epoch_log_params['revenue'] = \
                self.env.get_revenue(self.env)

        # plotting
        if epoch % self.logging.plot_frequency == 0:
            print("\tcurrent utilities: " + str(self._cur_epoch_log_params['utilities'].tolist()))

            unique_bidders = [i[0] for i in self._model2bidder]
            # TODO: possibly want to use old valuations, but currently it uses
            #       those from the util_loss, not those that were used during self-play
            o = torch.stack(
                [self.env._observations[:self.plot_points, b, ...] for b in unique_bidders],
                dim=1
            )
            b = torch.stack([self.env.agents[b[0]].get_action(o[:, i, ...])
                             for i, b in enumerate(self._model2bidder)], dim=1)

            labels = ['NPGA agent {}'.format(i) for i in range(len(self.models))]
            fmts = ['o'] * len(self.models)
            if self.known_bne and self.logging.log_metrics['opt']:
                for env_idx, _ in enumerate(self.bne_env):
                    o = torch.cat([o, self.v_opt[env_idx]], dim=1)
                    b = torch.cat([b, self.b_opt[env_idx]], dim=1)
                    labels += [f"BNE{'_' + str(env_idx + 1) if len(self.bne_env) > 1 else ''} agent {j}"
                               for j in range(len(self.models))]
                    fmts += ['--'] * len(self.models)

            self._plot(plot_data=(o, b), writer=self.writer, figure_name='bid_function',
                       epoch=epoch, labels=labels, fmts=fmts, plot_points=self.plot_points)

        self.overhead = self.overhead + timer() - start_time
        self._cur_epoch_log_params['overhead_hours'] = self.overhead / 3600
        if self.writer:
            self.writer.add_metrics_dict(
                self._cur_epoch_log_params, self.model_names, epoch,
                group_prefix=None, metric_tag_mapping = metrics.MAPPING_METRICS_TAGS)
        elapsed_overhead = timer() - start_time
        print('epoch {}:\toverhead {:.3f}s'.format(epoch, elapsed_overhead), end="\r")    


    def exit_run(self, global_step=None):
        if self.logging.enable_logging:
            self._log_experiment_params(global_step=global_step)

        if self.logging.save_models:
            self._save_models(directory=self.run_log_dir)

        del self.writer  # make this explicit to force cleanup and closing of tb-logfiles
        self.writer = None


    def _setup_plot_equilibirum_data(self):
        # set up list for (multiple) BNE valuations and bids
        self.v_opt = [None] * len(self.bne_env)
        self.b_opt = [None] * len(self.bne_env)

        # Switch needed for high dimensional settings, where we can't
        # exactly match the requested grid (see e.g. multi-unit simplex)
        grid_size_differs = False

        # Draw valuations and corresponding equilibrium bids in all the
        # availabe BNE
        for bne_id, bne_env in enumerate(self.bne_env):
            # dim: [points, models, valuation_size]
            # get one representative player for each model
            model_players = [m[0] for m in self._model2bidder]
            self.v_opt[bne_id] = torch.stack(
                [self.sampler.generate_valuation_grid(i, self.plot_points)
                 for i in model_players],
                dim=1)
            self.b_opt[bne_id] = torch.stack(
                [self._optimal_bid[bne_id](
                    self.v_opt[bne_id][:, model_id, :],
                    player_position=model_players[model_id])
                 for model_id in range(len(model_players))],
                dim=1)
            if self.v_opt[bne_id].shape[0] != self.plot_points:
                grid_size_differs = True

        if grid_size_differs:
            print('´plot_points´ changed due to get_valuation_grid')
            self.plot_points = self.v_opt[0].shape[0]



    def _calculate_metrics_known_bne(self):
        """
        Compare performance to BNE and return:
            utility_vs_bne: List[Tensor] of length `len(self.bne_env)`, length of Tensor `n_models`.
            epsilon_relative: List[Tensor] of length `len(self.bne_env)`, length of Tensor `n_models`.
            epsilon_absolute: List[Tensor] of length `len(self.bne_env)`, length of Tensor `n_models`.

        These are all lists of lists. The outer list corrsponds to which BNE is comapred
        (usually there's only one BNE). Each inner list is of length `self.n_models`.
        """
        # shorthand for model to bidder index conversion
        m2b = lambda m: self._model2bidder[m][0]

        utility_vs_bne = [None] * len(self.bne_env)
        epsilon_relative = [None] * len(self.bne_env)
        epsilon_absolute = [None] * len(self.bne_env)

        for bne_idx, bne_env in enumerate(self.bne_env):
            # generally redraw bne_vals, except when this is expensive!
            # length: n_models
            # TODO Stefan: this seems to be false in most settings, even when not desired.
            redraw_bne_vals = not self.logging.cache_eval_actions
            # length: n_models
            utility_vs_bne[bne_idx] = torch.tensor([
                bne_env.get_strategy_reward(
                    strategy=model,
                    player_position=m2b(m),
                    redraw_valuations=redraw_bne_vals
                ) for m, model in enumerate(self.models)
            ])
            epsilon_relative[bne_idx] = torch.tensor(
                [1 - utility_vs_bne[bne_idx][i] / self.bne_utilities[bne_idx][m2b(i)]
                 for i, model in enumerate(self.models)]
            )
            epsilon_absolute[bne_idx] = torch.tensor(
                [self.bne_utilities[bne_idx][m2b(i)] - utility_vs_bne[bne_idx][i]
                 for i, model in enumerate(self.models)]
            )

        return utility_vs_bne, epsilon_relative, epsilon_absolute


    def _calculate_metrics_action_space_norms(self):
        """
        Calculate "action space distance" of model and bne-strategy. If
        `self.logging.log_componentwise_norm` is set to true, will only
        return norm of the best action dimension.

        Returns:
            L_2 and L_inf: each a List[Tensor] of length `len(self.bne_env)`, length of Tensor `n_models`.
        """

        L_2 = [None] * len(self.bne_env)
        L_inf = [None] * len(self.bne_env)
        for bne_idx, bne_env in enumerate(self.bne_env):
            # shorthand for model to agent

            # we are only using m2a locally within this loop, so we can safely ignore the following pylint warning:
            # pylint: disable=cell-var-from-loop

            m2a = lambda m: bne_env.agents[self._model2bidder[m][0]]
            m2o = lambda m: bne_env._observations[:, self._model2bidder[m][0], :]

            L_2[bne_idx] = torch.tensor([
                metrics.norm_strategy_and_actions(
                    strategy=model,
                    actions=m2a(i).get_action(),
                    valuations=m2o(i),
                    p=2,
                    componentwise=self.logging.log_componentwise_norm
                )
                for i, model in enumerate(self.models)
            ])
            L_inf[bne_idx] = torch.tensor([
                metrics.norm_strategy_and_actions(
                    strategy=model,
                    actions=m2a(i).get_action(),
                    valuations=m2o(i),
                    p=float('inf'),
                    componentwise=self.logging.log_componentwise_norm
                )
                for i, model in enumerate(self.models)
            ])
        return L_2, L_inf

    def _calculate_metrics_util_loss(self, create_plot_output: bool, epoch: int = None,
                                     batch_size=None, grid_size=None):
        """
        Compute mean util_loss of current policy and return ex interim util
        loss (ex ante util_loss is the average of that tensor).

        Returns:
            ex_ante_util_loss: List[torch.tensor] of length self.n_models
            ex_interim_max_util_loss: List[torch.tensor] of length self.n_models
        """

        env = self.env
        if batch_size is None:
            # take min of both in case the requested batch size is too large for the env
            batch_size = min(self.logging.util_loss_batch_size, self.learning.batch_size)
        if grid_size is None:
            grid_size = self.logging.util_loss_grid_size

        assert batch_size <= env.batch_size, \
            "Util_loss for larger than actual batch size not implemented."

        with torch.no_grad():  # don't need any gradient information here
            # TODO: currently we don't know where exactly a mmory leak is
            observations = self.env._observations[:batch_size, :, :]
            util_losses, best_responses = zip(*[
                metrics.ex_interim_util_loss(
                    env=env,
                    player_position=player_positions[0],
                    agent_observations=observations[:, player_positions[0], :],
                    grid_size=grid_size
                )
                for player_positions in self._model2bidder
            ])

        if self.logging.best_response:
            plot_data = (observations[:, [b[0] for b in self._model2bidder], :],
                         torch.stack(best_responses, 1))
            labels = ['NPGA_{}'.format(i) for i in range(len(self.models))]
            fmts = ['o'] * len(self.models)
            self._plot(plot_data=plot_data, writer=self.writer,
                       ylim=[0, self.sampler.support_bounds.max().item()],
                       figure_name='best_responses', y_label='best response',
                       colors=list(range(len(self.models))), epoch=epoch,
                       labels=labels, fmts=fmts, plot_points=self.plot_points)

        # calculate different losses
        ex_ante_util_loss = [util_loss_model.mean() for util_loss_model in util_losses]
        ex_interim_max_util_loss = [util_loss_model.max() for util_loss_model in util_losses]
        estimated_relative_ex_ante_util_loss = [
            (1 - u / (u + l)).item()
            for u, l in zip(
                [self.env.get_reward(self.env.agents[self._model2bidder[m][0]]).detach()
                for m in range(len(self.models))],
                ex_ante_util_loss)
        ]

        # plotting
        if create_plot_output:

            # keep track of upper bound for plotting
            if not hasattr(self, '_max_util_loss'):
                self._max_util_loss = ex_interim_max_util_loss

            # Transform to output with dim(batch_size, n_models, n_bundle), for util_losses n_bundle=1
            util_losses = torch.stack(list(util_losses), dim=1).unsqueeze_(-1)
            observations = self.env._observations[:batch_size, :, :]
            plot_data = (observations[:, [b[0] for b in self._model2bidder], :], util_losses)
            labels = ['NPGA_{}'.format(i) for i in range(len(self.models))]
            fmts = ['o'] * len(self.models)
            self._plot(plot_data=plot_data, writer=self.writer,
                       ylim=[0, max(self._max_util_loss).detach().item()],
                       figure_name='util_loss_landscape', y_label='ex-interim loss',
                       colors=list(range(len(self.models))), epoch=epoch,
                       labels=labels, fmts=fmts, plot_points=self.plot_points)

        return ex_ante_util_loss, ex_interim_max_util_loss, estimated_relative_ex_ante_util_loss

    def _log_experiment_params(self, global_step=None):
        """Logging of paramters after learning finished.

        Arguments:
            global_step, int: number of completed iterations/epochs. Will usually
                be equal to `self.running.n_epochs`

        Returns:
            Writes to `self.writer`.

        """
        # TODO: write out all experiment params (complete dict) #See issue #113
        # TODO: Stefan: this currently called _per run_. is this desired behavior?
        for i, model in enumerate(self.models):
            self.writer.add_text('hyperparameters/neural_net_spec', str(model))
            self.writer.add_graph(model, self.env._observations[:, i, :])

        h_params = {'hyperparameters/batch_size': self.learning.batch_size,
                    'hyperparameters/pretrain_iters': self.learning.pretrain_iters,
                    'hyperparameters/hidden_nodes': str(self.learning.hidden_nodes),
                    'hyperparameters/hidden_activations': str(self.learning.hidden_activations),
                    'hyperparameters/optimizer_hyperparams': str(self.learning.optimizer_hyperparams),
                    'hyperparameters/optimizer_type': self.learning.optimizer_type}

        ignored_metrics = ['utilities', 'update_norm', 'overhead_hours']
        filtered_metrics = filter(lambda elem: elem[0] not in ignored_metrics,
                                  self._cur_epoch_log_params.items())
        try:
            for k, v in filtered_metrics:
                if isinstance(v, (list, torch.Tensor)):
                    for model_number, metric in enumerate(v):
                        self._hparams_metrics["metrics/" + k+"_"+str(model_number)] = metric
                elif isinstance(v, int) or isinstance(v, float):
                    self._hparams_metrics["metrics/" + k] = v
                else:
                    print("the type ", type(v), " is not supported as a metric")
        except Exception as e:  # pylint: disable=broad-except
            print(e)
        self.writer.add_hparams(hparam_dict=h_params, metric_dict=self._hparams_metrics,
                                global_step=global_step)

    def _save_models(self, directory):
        # TODO: maybe we should also log out all pointwise util_losses in the ending-epoch to disk to
        # use it to make nicer plots for a publication? --> will be done elsewhere. Logging. Assigned to @Hlib/@Stefan
        for model, player_position in zip(self.models, self._model2bidder):
            name = 'model_' + str(player_position[0]) + '.pt'
            torch.save(model.state_dict(), os.path.join(directory, 'models', name))

    def export_stepwise_linear_bid(self, bidders: List[Bidder]):
        """
        exporting grid valuations and corresponding bids for usage of verifier.

        Args
        ----
            experiment_dir: str, dir where export is going to be saved
            bidders: List[Bidder], to be evaluated here
            step: float, step length

        Returns
        -------
            to disk: List[csv]
        """
        step = self.logging.export_step_wise_linear_bid_function_size
        for bidder in bidders:
            val = bidder.get_valuation_grid(n_points=None, step=step,
                                            dtype=torch.float64, extended_valuation_grid=True)
            bid = bidder.strategy.forward(val.to(torch.float32)).to(torch.float64)            
            cat = torch.cat((val, bid), axis=1)
            file_dir = self.run_log_dir + '/bidder_' + str(bidder.player_position) + '_export.csv'
            np.savetxt(file_dir, cat.detach().cpu().numpy(), fmt='%1.16f', delimiter=",")


    _full_log_file_name = 'full_results'
    _aggregate_log_file_name = 'aggregate_log'
    _configurations_f_name = 'experiment_configurations.json'
    _git_commit_hash_file_name = 'git_hash'

    # based on https://stackoverflow.com/a/57411105/4755970
    # experiment must be the directory immediately above the runs and each run must have the same shape.
    # No aggregation of multiple subdirectories for now.
    def tabulate_tensorboard_logs(self):
        """
        This function reads all tensorboard event log files in subdirectories and converts their content into
        a single csv file containing info of all runs.
        """
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator, STORE_EVERYTHING_SIZE_GUIDANCE        

        if not self.logging.save_tb_events_to_csv_detailed and not self.logging.save_tb_events_to_csv_aggregate and not \
            self.logging.save_tb_events_to_binary_detailed:
            return
        print('Tabulating tensorboard logs...', end=' ')
        # runs are all subdirectories that don't start with '.' (exclude '.ipython_checkpoints')
        # add more filters as needed
        runs = [x.name for x in os.scandir(self.experiment_log_dir) if
                x.is_dir() and not x.name.startswith('.') and not x.name == 'alternative']

        all_tb_events = {'run': [], 'subrun': [], 'tag': [], 'epoch': [], 'value': [], 'wall_time': []}
        last_epoch_tb_events = {'run': [], 'subrun': [], 'tag': [], 'epoch': [], 'value': [], 'wall_time': []}
        for run in runs:
            subruns = [x.name for x in os.scandir(os.path.join(self.experiment_log_dir, run))
                    if x.is_dir() and any(file.startswith('events.out.tfevents')
                                            for file in os.listdir(os.path.join(self.experiment_log_dir, run, x.name)))]
            subruns.append('.')  # also read global logs
            for subrun in subruns:
                ea = EventAccumulator(os.path.join(self.experiment_log_dir, run, subrun),
                                    size_guidance=STORE_EVERYTHING_SIZE_GUIDANCE).Reload()

                tags = ea.Tags()['scalars']

                for tag in tags:
                    for event in ea.Scalars(tag):
                        all_tb_events['run'].append(run)
                        all_tb_events['subrun'].append(subrun)
                        all_tb_events['tag'].append(tag)
                        all_tb_events['value'].append(event.value)
                        all_tb_events['wall_time'].append(event.wall_time)
                        all_tb_events['epoch'].append(event.step)

                    last_epoch_tb_events['run'].append(run)
                    last_epoch_tb_events['subrun'].append(subrun)
                    last_epoch_tb_events['tag'].append(tag)
                    # a last event is always guaranteed to exist, we can ignore pylint's warning
                    # pylint: disable=undefined-loop-variable
                    last_epoch_tb_events['value'].append(event.value)
                    last_epoch_tb_events['wall_time'].append(event.wall_time)
                    last_epoch_tb_events['epoch'].append(event.step)

        all_tb_events = pd.DataFrame(all_tb_events)
        last_epoch_tb_events = pd.DataFrame(last_epoch_tb_events)

        if self.logging.save_tb_events_to_csv_detailed:
            f_name = os.path.join(self.experiment_log_dir, f'{Logger._full_log_file_name}.csv')
            all_tb_events.to_csv(f_name, index=False)

        if self.logging.save_tb_events_to_csv_aggregate:
            f_name = os.path.join(self.experiment_log_dir, f'{Logger._aggregate_log_file_name}.csv')
            last_epoch_tb_events.to_csv(f_name, index=False)

        if self.logging.save_tb_events_to_binary_detailed:
            f_name = os.path.join(self.experiment_log_dir, f'{Logger._full_log_file_name}.pkl')
            all_tb_events.to_pickle(f_name)
        
        # print_aggregate_tensorboard_logs(self.experiment_log_dir)
        print('finished tabulating logs.')


    def process_figure(self, fig, epoch=None, figure_name='plot', tb_group='eval', tb_writer=None):
        """displays, logs and/or saves a figure"""
        display=self.logging.plot_show_inline
        save_png=self.logging.save_figure_to_disk_png
        save_svg=self.logging.save_figure_to_disk_svg
        output_dir = self.run_log_dir

        if save_png and output_dir:
            plt.savefig(os.path.join(output_dir, 'png', f'{figure_name}_{epoch:05}.png'))

        if save_svg and output_dir:
            plt.savefig(os.path.join(output_dir, 'svg', f'{figure_name}_{epoch:05}.svg'),
                        format='svg', dpi=1200)
        if tb_writer:
            tb_writer.add_figure(f'{tb_group}/{figure_name}', fig, epoch)

        if display:
            plt.show()

    def log_git_commit_hash(self):
        """Saves the hash of the current git commit into experiment_dir."""
        import subprocess
        # Will leave it here as a comment in case we'll ever need to log the full dependency tree or the environment.
        # os.system('pipdeptree --json-tree > dependencies.json')
        # os.system('conda env export > environment.yml')

        try:
            commit_hash = str(subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip())[2:-1]
            with open(os.path.join(self.experiment_log_dir, f'{Logger._git_commit_hash_file_name}.txt'), "w") as text_file:
                text_file.write(commit_hash)
        except Exception as e:
            warnings.warn("Failed to retrieve and log the git commit hash.")

    @staticmethod
    def print_full_tensorboard_logs(experiment_dir, first_row: int = 0, last_row=None):
        """
        Prints in a tabular form the full log from all the runs in the current experiment, reads data from a pkl file
        in the experiment directory
        :param first_row: the first row to be printed if the full log is used
        :param last_row: the last row to be printed if the full log is used
        """
        f_name = os.path.join(experiment_dir, f'{_full_log_file_name}.pkl')
        objects = []
        with (open(f_name, "rb")) as full_results:
            while True:
                try:
                    objects.append(pickle.load(full_results))
                except EOFError:
                    break
        if last_row is None:
            last_row = len(objects[0])
        print('Full log:')
        print(objects[0].iloc[first_row:last_row].to_markdown())

    @staticmethod
    def print_aggregate_tensorboard_logs(experiment_dir):
        """
        Prints in a tabular form the aggregate log from all the runs in the current experiment,
        reads data from the csv file in the experiment directory
        """
        f_name = os.path.join(experiment_dir, f'{Logger._aggregate_log_file_name}.csv')
        df = pd.read_csv(f_name)
        print('Aggregate log:')
        print(df.to_markdown())

    @staticmethod
    def save_experiment_config(experiment_log_dir, experiment_configuration: ExperimentConfig):
        """
        Serializes ExperimentConfiguration into a readable JSON file

        :param experiment_log_dir: full path except for the file name
        :param experiment_configuration: experiment configuration as given by ConfigurationManager
        """
        
        f_name = os.path.join(experiment_log_dir, Logger._configurations_f_name)

        temp_cp = experiment_configuration.setting.common_prior
        temp_ha = experiment_configuration.learning.hidden_activations

        experiment_configuration.setting.common_prior = str(experiment_configuration.setting.common_prior)
        experiment_configuration.learning.hidden_activations = str(
            experiment_configuration.learning.hidden_activations)
        with open(f_name, 'w+') as outfile:
            json.dump(experiment_configuration, outfile, cls=EnhancedJSONEncoder, indent=4)

        # Doesn't look so shiny, but probably the quickest way to prevent compromising the object
        experiment_configuration.setting.common_prior = temp_cp
        experiment_configuration.learning.hidden_activations = temp_ha
