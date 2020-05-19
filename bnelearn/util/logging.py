"""This module contains utilities for logging of experiments"""

import os
import pickle
import time
import warnings
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard.summary import hparams
from torch.utils.tensorboard.writer import FileWriter, SummaryWriter, scalar

_full_log_file_name = 'full_results'
_aggregate_log_file_name = 'aggregate_log'


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


class CustomSummaryWriter(SummaryWriter):
    """
    Extends SummaryWriter with a method to add multiple scalars in the way
    that we intend. The original SummaryWriter can either add a single scalar at a time
    or multiple scalars, but in the latter case, multiple runs are created without
    the option to control these.
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

        # Simply commenting the code below out would end up in registering no value for the metric in the hparams tab
        # (doesn't seem like values are matched automatically in any way)
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
