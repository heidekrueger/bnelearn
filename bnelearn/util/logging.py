"""This module contains utilities for logging of experiments"""
import os
import pickle
import subprocess
import time
from typing import List
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator, STORE_EVERYTHING_SIZE_GUIDANCE
from torch.utils.tensorboard.summary import hparams
from torch.utils.tensorboard.writer import FileWriter, SummaryWriter, scalar
import pkg_resources

from bnelearn.bidder import Bidder
from bnelearn.experiment.configurations import *


_full_log_file_name = 'full_results'
_aggregate_log_file_name = 'aggregate_log'
_configurations_f_name = 'experiment_configurations.json'
_git_commit_hash_file_name = 'git_hash'


# based on https://stackoverflow.com/a/57411105/4755970
# experiment must be the directory immediately above the runs and each run must have the same shape.
# No aggregation of multiple subdirectories for now.
def tabulate_tensorboard_logs(experiment_dir, write_aggregate=True, write_detailed=False, write_binary=False):
    """
    This function reads all tensorboard event log files in subdirectories and converts their content into
    a single csv file containing info of all runs.
    """
    # runs are all subdirectories that don't start with '.' (exclude '.ipython_checkpoints')
    # add more filters as needed
    runs = [x.name for x in os.scandir(experiment_dir) if
            x.is_dir() and not x.name.startswith('.') and not x.name == 'alternative']

    all_tb_events = {'run': [], 'subrun': [], 'tag': [], 'epoch': [], 'value': [], 'wall_time': []}
    last_epoch_tb_events = {'run': [], 'subrun': [], 'tag': [], 'epoch': [], 'value': [], 'wall_time': []}
    for run in runs:
        subruns = [x.name for x in os.scandir(os.path.join(experiment_dir, run))
                   if x.is_dir() and any(file.startswith('events.out.tfevents')
                                         for file in os.listdir(os.path.join(experiment_dir, run, x.name)))]
        subruns.append('.')  # also read global logs
        for subrun in subruns:
            ea = EventAccumulator(os.path.join(experiment_dir, run, subrun),
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

    if write_detailed:
        f_name = os.path.join(experiment_dir, f'{_full_log_file_name}.csv')
        all_tb_events.to_csv(f_name, index=False)

    if write_aggregate:
        f_name = os.path.join(experiment_dir, f'{_aggregate_log_file_name}.csv')
        last_epoch_tb_events.to_csv(f_name, index=False)

    if write_binary:
        f_name = os.path.join(experiment_dir, f'{_full_log_file_name}.pkl')
        all_tb_events.to_pickle(f_name)


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


def print_aggregate_tensorboard_logs(experiment_dir):
    """
    Prints in a tabular form the aggregate log from all the runs in the current experiment,
    reads data from the csv file in the experiment directory
    """
    f_name = os.path.join(experiment_dir, f'{_aggregate_log_file_name}.csv')
    df = pd.read_csv(f_name)
    print('Aggregate log:')
    print(df.to_markdown())


def log_git_commit_hash(experiment_dir):
    """Saves the hash of the current git commit into experiment_dir."""

    # Will leave it here as a comment in case we'll ever need to log the full dependency tree or the environment.
    # os.system('pipdeptree --json-tree > dependencies.json')
    # os.system('conda env export > environment.yml')

    try:
        commit_hash = str(subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip())[2:-1]
        with open(os.path.join(experiment_dir, f'{_git_commit_hash_file_name}.txt'), "w") as text_file:
            text_file.write(commit_hash)
    except Exception as e:
        warnings.warn("Failed to retrieve and log the git commit hash.")


def save_experiment_config(experiment_log_dir, experiment_configuration: ExperimentConfig):
    """
    Serializes ExperimentConfiguration into a readable JSON file

    :param experiment_log_dir: full path except for the file name
    :param experiment_configuration: experiment configuration as given by ConfigurationManager
    """
    f_name = os.path.join(experiment_log_dir, _configurations_f_name)

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


def process_figure(fig, epoch=None, figure_name='plot', tb_group='eval',
                   tb_writer=None, display=False,
                   output_dir=None, save_png=False, save_svg=False):
    """displays, logs and/or saves a figure"""

    if save_png and output_dir:
        plt.savefig(os.path.join(output_dir, 'png', f'{figure_name}_{epoch:05}.png'))

    if save_svg and output_dir:
        plt.savefig(os.path.join(output_dir, 'svg', f'{figure_name}_{epoch:05}.svg'),
                    format='svg', dpi=1200)
    if tb_writer:
        tb_writer.add_figure(f'{tb_group}/{figure_name}', fig, epoch)

    if display:
        plt.show()


def export_stepwise_linear_bid(experiment_dir, bidders: List[Bidder], step=1e-2):
    """
    expoerting grid valuations and corresponding bids for usage of verifier.

    Args
    ----
        experiment_dir: str, dir where export is going to be saved
        bidders: List[Bidder], to be evaluated here
        step: float, step length

    Returns
    -------
        to disk: List[csv]
    """
    for bidder in bidders:
        val = bidder.get_valuation_grid(n_points=None, step=step,
                                        dtype=torch.float64, extended_valuation_grid=True)
        bid = bidder.strategy.forward(val.to(torch.float32)).to(torch.float64)
        cat = torch.cat((val, bid), axis=1)
        file_dir = experiment_dir + '/bidder_' + str(bidder.player_position) + '_export.csv'
        np.savetxt(file_dir, cat.detach().cpu().numpy(), fmt='%1.16f', delimiter=",")


class CustomSummaryWriter(SummaryWriter):
    """
    Extends SummaryWriter with two methods:

    * a method to add multiple scalars in the way that we intend. The original
        SummaryWriter can either add a single scalar at a time or multiple scalars,
        but in the latter case, multiple runs are created without
        the option to control these.
    * overwriting the the add_hparams method to write hparams without creating
        another tensorboard run file
    """

    def add_hparams(self, hparam_dict=None, metric_dict=None, global_step=None):
        """
        Overides the parent method to prevent the creation of unwanted additional subruns while logging hyperparams,
        as it is done by the original PyTorch method
        """
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict)


        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)

        for k, v in metric_dict.items():
            self.add_scalar(k, v, global_step=global_step)

    def add_metrics_dict(self, metrics_dict: dict, run_suffices: List[str],
                         global_step=None, walltime=None,
                         group_prefix: str = None):
        """
        Args:
            metric_dict (dict): A dict of metrics. Keys are tag names, values are values.
                values can be float, List[float] or Tensor.
                When List or (nonscalar) tensor, the length must match n_models
            run_suffices (List[str]): if each value in metrics_dict is scalar, doesn't need to be supplied.
                When metrics_dict contains lists/iterables, they must all have the same length which should be equal to
                the length of run_suffices
        """
        torch._C._log_api_usage_once("tensorboard.logging.add_scalar")
        walltime = time.time() if walltime is None else walltime
        fw_logdir = self._get_file_writer().get_logdir()

        if run_suffices is None:
            run_suffices = []

        l = len(run_suffices)

        for key, vals in metrics_dict.items():
            tag = key if not group_prefix else group_prefix + '/' + key

            if isinstance(vals, float) or isinstance(vals, int) or (
                    torch.is_tensor(vals) and vals.size() in {torch.Size([]), torch.Size([1])}):
                # Only a single value --> log directly in main run
                self.add_scalar(tag, vals, global_step, walltime)
            elif len(vals) == 1:
                # List type of length 1, but not tensor --> extract item
                self.add_scalar(tag, vals[0], global_step, walltime)
            elif len(vals) == l:
                # Log each into a run with its own prefix.
                for suffix, scalar_value in zip(run_suffices, vals):
                    fw_tag = fw_logdir + "/" + suffix.replace("/", "_")

                    if fw_tag in self.all_writers.keys():
                        fw = self.all_writers[fw_tag]
                    else:
                        fw = FileWriter(fw_tag, self.max_queue, self.flush_secs,
                                        self.filename_suffix)
                        self.all_writers[fw_tag] = fw
                    # Not using caffe2 -->following line is commented out from original SummaryWriter implementation
                    # if self._check_caffe2_blob(scalar_value):
                    #     scalar_value = workspace.FetchBlob(scalar_value)
                    fw.add_summary(scalar(tag, scalar_value), global_step, walltime)
            else:
                raise ValueError('Got list of invalid length.')


def read_bne_utility_database(exp: 'Experiment'):
    """Check if this setting's BNE has been saved to disk before.

    Args:
        exp: Experiment

    Returns:
        db_batch_size: int
            sample size of a DB entry if found, else -1
        db_bne_utility: List[n_players]
            list of the saved BNE utilites
    """
    file_path = pkg_resources.resource_filename(__name__, 'bne_database.csv')
    bne_database = pd.read_csv(file_path)

    # see if we already have a sample
    setting_database = bne_database[
        (bne_database.experiment_class == str(type(exp))) &
        (bne_database.payment_rule == exp.payment_rule) &
        (bne_database.correlation == exp.correlation) &
        (bne_database.risk == exp.risk)
    ]

    # 1. no entry found for this exp
    if len(setting_database) == 0:
        return -1, None

    # 2. found entry
    else:
        return setting_database['batch_size'].tolist()[0], setting_database.bne_utilities.tolist()


def write_bne_utility_database(exp: 'Experiment', bne_utilities_sampled: list):
    """Write the sampled BNE utilities to disk.

    Args:
        exp: Experiment
        bne_utilities_sampled: list
            BNE utilites that are to be writen to disk
    """
    file_path = pkg_resources.resource_filename(__name__, 'bne_database.csv')
    bne_database = pd.read_csv(file_path)

    bne_env = exp.bne_env if not isinstance(exp.bne_env, list) \
        else exp.bne_env[0]

    # see if we already have a sample
    setting_database = bne_database[
        (bne_database.experiment_class == str(type(exp))) &
        (bne_database.payment_rule == exp.payment_rule) &
        (bne_database.correlation == exp.correlation) &
        (bne_database.risk == exp.risk)
    ]

    # No entry found: make new db entry
    if len(setting_database) == 0:
        for player_position in [agent.player_position for agent in exp.bne_env.agents]:
            bne_database = bne_database.append(
                {
                    'experiment_class': str(type(exp)),
                    'payment_rule':     exp.payment_rule,
                    'correlation':      exp.correlation,
                    'risk':             exp.risk,
                    'player_position':  player_position,
                    'batch_size':       bne_env.batch_size,
                    'bne_utilities':    bne_utilities_sampled[player_position].item()
                },
                ignore_index=True
            )

    # Overwrite database entry
    else:
        for player_position in [agent.player_position for agent in exp.bne_env.agents]:
            bne_database.loc[
                (bne_database.experiment_class == str(type(exp))) &
                (bne_database.payment_rule == exp.payment_rule) &
                (bne_database.correlation == exp.correlation) &
                (bne_database.risk == exp.risk) &
                (bne_database.player_position == player_position)
            ] = [[str(type(exp)), exp.payment_rule, exp.correlation, exp.risk,
                  player_position, bne_env.batch_size,
                  bne_utilities_sampled[player_position].item()]]

    bne_database.to_csv(file_path, index=False)
