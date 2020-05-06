"""This module contains utilities for logging of experiments"""

import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import warnings
import matplotlib.pyplot as plt

# based on https://stackoverflow.com/a/57411105/4755970
# output_dir must be the directory immediately above the runs and each run must have the same shape.
# No aggregation of multiple subdirectories for now.
def convert_tb_to_csv(directory, write_aggregate=True, write_detailed=False):
    """
    This function reads all tensorboard event log files in subdirectories and converts their content into
    a single csv file containing info of all runs.
    """
    output_dir = directory

    # runs are all subdirectories that don't start with '.' (exclude '.ipython_checkpoints')
    # add more filters as needed
    runs = [x.name for x in os.scandir(output_dir) if
            x.is_dir() and not x.name.startswith('.') and not x.name == 'alternative']

    d = {'run': [], 'tag': [], 'epoch': [], 'value': [], 'wall_time': []}
    for run in runs:
        ea = EventAccumulator(os.path.join(output_dir, run)).Reload()
        tags = ea.Tags()['scalars']

        for tag in tags:
            for event in ea.Scalars(tag):
                d['run'].append(run)
                d['tag'].append(tag)
                d['value'].append(event.value)
                d['wall_time'].append(event.wall_time)
                d['epoch'].append(event.step)

    df = pd.DataFrame(d)
    if write_detailed:
        df.to_csv(os.path.join(output_dir, f'full_results.csv'))

    if write_aggregate:
        warnings.warn('Data Frame aggregation not Implemented')


def process_figure(fig, epoch=None, figure_name='plot', tb_group='eval',
                   tb_writer=None, display = False, 
                   output_dir = None, save_png=False, save_svg=False):
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