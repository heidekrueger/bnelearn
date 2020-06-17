"""This module contains utilities for logging of experiments"""

import pickle
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard.summary import hparams
from torch.utils.tensorboard.writer import FileWriter, SummaryWriter, scalar
from bnelearn.bidder import Bidder

from bnelearn.experiment.configurations import *

_full_log_file_name = 'full_results'
_aggregate_log_file_name = 'aggregate_log'
_configurations_f_name = 'experiment_configurations.json'


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

            ea = EventAccumulator(os.path.join(experiment_dir, run, subrun)).Reload()
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
        val = bidder.get_valuation_grid(n_points=None, step=step, dtype=torch.float64)
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

    def add_hparams(self, hparam_dict=None, metric_dict=None):
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
            self.add_scalar(k, v)

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


def log_experiment_configurations(experiment_log_dir, experiment_configuration: ExperimentConfig):
    """
    Serializes ExperimentConfiguration into a readable JSON file

    :param experiment_log_dir: full path except for the file name
    :param experiment_configuration: experiment configuration as given by ConfigurationManager
    """
    f_name = os.path.join(experiment_log_dir, _configurations_f_name)

    experiment_configuration.setting.common_prior = str(experiment_configuration.setting.common_prior)
    experiment_configuration.learning.hidden_activations = str(
        experiment_configuration.learning.hidden_activations)
    with open(f_name, 'w+') as outfile:
        json.dump(experiment_configuration, outfile, cls=EnhancedJSONEncoder, indent=4)


def get_experiment_config_from_configurations_log(experiment_log_dir=None):
    """
    Retrieves stored configurations from JSON and turns them into ExperimentConfiguration object
    By default creates configuration from the file stored alongside the running script

    :param experiment_log_dir: full path except for the file name, current working directory by default
    :return: ExperimentConfiguration object
    """
    if experiment_log_dir is None:
            experiment_log_dir = os.path.abspath(os.getcwd())
    f_name = os.path.join(experiment_log_dir, _configurations_f_name)

    with open(f_name) as json_file:
        experiment_config_as_dict = json.load(json_file)

    experiment_config = ExperimentConfig(experiment_class=experiment_config_as_dict['experiment_class'])

    config_set_name_to_obj = {
        'running': RunningConfig(),
        'setting': SettingConfig(),
        'learning': LearningConfig(),
        'logging': LoggingConfig(),
        'hardware': HardwareConfig()
    }

    # Parse a dictionary retrieved from JSON into ExperimentConfiguration object
    # Attribute assignment pattern: experiment_config.config_group_name.config_group_object_attr = attr_val
    # e.g. experiment_config.run_config.n_runs = experiment_config_as_dict['run_config']['n_runs']
    # config_group_object assignment pattern: experiment_config.config_group_name = config_group_object
    # e.g. experiment_config.run_config = earlier initialised and filled instance of RunningConfiguration class
    experiment_config_as_dict = {k: v for (k, v) in experiment_config_as_dict.items() if k != 'experiment_class'}.items()
    for config_set_name, config_group_dict in experiment_config_as_dict:
        for config_set_obj_attr, attr_val in config_group_dict.items():
            setattr(config_set_name_to_obj[config_set_name], config_set_obj_attr, attr_val)
        setattr(experiment_config, config_set_name, config_set_name_to_obj[config_set_name])

    # Create hidden activations object based on the loaded string
    hidden_activations_methods = {'SELU': lambda: nn.SELU}
    ha = str(experiment_config.learning.hidden_activations).split('()')
    for symb in ['[', ']', ' ', ',']:
        ha = list(map(lambda s: str(s).replace(symb, ''), ha))
    ha = [i for i in ha if i != '']
    ha = [hidden_activations_methods[layer]()() for layer in ha]
    experiment_config.learning.hidden_activations = ha

    # Create common_prior object based on the loaded string
    common_priors = {'Uniform': torch.distributions.uniform.Uniform}
    distribution = str(experiment_config.setting.common_prior).split('(')[0]
    experiment_config.setting.common_prior = common_priors[distribution](low=experiment_config.setting.u_lo,
                                                                         high=experiment_config.setting.u_hi)

    return experiment_config
