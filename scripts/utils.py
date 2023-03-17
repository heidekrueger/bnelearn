"""Some helper functions for collecting and summarizing logging data.

WARNING: This file is not maintained.
"""
import os, sys
import re
import torch
import pandas as pd
import numpy as np
import json
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib.pyplot as plt
import json
markers = ['o', '^', 's', 'p', '.', '+']
colors = [(0/255.,191/255.,196/255.), (248/255.,118/255.,109/255.),
          (150/255.,120/255.,170/255.), (255/255.,215/255.,130/255.)] 

sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'bnelearn'))

from bnelearn.strategy import NeuralNetStrategy
from bnelearn.experiment.configuration_manager import ConfigurationManager
from bnelearn.util import logging
from bnelearn.util.metrics import ALIASES_LATEX


def multiple_exps_logs_to_df(
        path: str or dict,
        metrics: list = ['eval/L_2', 'eval/epsilon_relative', 'eval/util_loss_ex_interim',
                         'eval/estimated_relative_ex_ante_util_loss',
                         'eval/efficiency', 'eval/revenue', 'eval/utilities'],
        precision: int = 4,
        with_stddev: bool = False,
        setting_parameters: list = ['correlation_types', 'correlation_coefficients',
                                    'risk', 'payment_rule', 'n_items', 'n_players'],
        save: bool = False
    ):
    """Creates and returns a Pandas DataFrame from all logs in `path`.

    This function is universally usable.

    Arguments:
        path: str or dict, which path to crawl for csv logs.
        metrics: list of which metrics we want to load in the df.
        precision: int of how many decimals we request.
        with_stddev: bool.
        setting_parameters: list, some hyperparams can be read in from
            the path itself. Turning this switch on pareses these values to
            individual columns.

    Returns:
        aggregate_df pandas Dataframe with one run corresponding to one row,
            columns correspond to the logged metrics (from the last iter).

    """
    if isinstance(path, str):
        experiments = [os.path.join(dp, f) for dp, dn, filenames
                       in os.walk(path) for f in filenames
                       if os.path.splitext(f)[1] == '.csv']
        experiments = {str(e): e for e in experiments}
    else:
        experiments = path

    form = '{:.' + str(precision) + 'f}'

    columns = ['Auction game'] + metrics + setting_parameters
    aggregate_df = pd.DataFrame(columns=columns)
    for exp_name, exp_path in experiments.items():
        df = pd.read_csv(exp_path)
        end_epoch = df.epoch.max()
        df = df[df.epoch == end_epoch]

        # load learning logs
        single_df = df.groupby(['tag'], as_index=False) \
            .agg({'value': ['mean', 'std']})
        single_df.columns = ['metric', 'mean','std']
        single_df = single_df.loc[single_df['metric'].isin(metrics)]

        def map_mean_std(row):
            result = str(form.format(round(row['mean'], precision)))
            if with_stddev:
                result += ' (' + str(form.format(round(row['std'], precision))) \
                    + ')'
            return result
        single_df[exp_name] = single_df.apply(map_mean_std, axis=1)

        single_df.index = single_df['metric']
        del single_df['mean'], single_df['std'], single_df['metric']
        single_df = single_df.T

        # load setting parameters
        try:
            with open(path + '/experiment_configurations.json') as json_file:
                config = json.load(json_file)
        except:
            with open(exp_path[:exp_path.rfind('/')] + '/experiment_configurations.json') as json_file:
                config = json.load(json_file)

        for param in setting_parameters:
            single_df[param] = config['setting'][param]

        aggregate_df = pd.concat([aggregate_df, single_df])
        aggregate_df['Auction game'][-1] = exp_name

    aggregate_df.columns = aggregate_df.columns.map(
        lambda m: ALIASES_LATEX[m] if m in ALIASES_LATEX.keys() else m
    )

    # write to file
    if save:
        aggregate_df.to_csv('experiments/summary.csv', index=False)

    return aggregate_df


def single_asym_exp_logs_to_df(
        exp_path: str or dict,
        metrics: list = ['eval/L_2', 'eval/epsilon_relative',
                         'eval/estimated_relative_ex_ante_util_loss'],
        precision: int = 4,
        with_stddev: bool = True,
        bidder_names: list = None
    ):
    """Creates and returns a Pandas DataFrame from the logs in `path` for an
    individual experiment with different bidders.

    This function is universally usable.

    Arguments:
        exp_path: str to `full_results.csv`.
        metrics: list of which metrics we want to load in the df.
        precision: int of how many decimals we request.
        with_stddev: bool.

    Returns:
        aggregate_df pandas Dataframe with one run corresponding to one row,
            columns correspond to the logged metrics (from the last iter).

    """
    form = '{:.' + str(precision) + 'f}'

    df = pd.read_csv(exp_path)
    end_epoch = df.epoch.max()
    df = df[df.epoch == end_epoch]

    df = df.groupby(
        ['tag', 'subrun'], as_index=False
    ).agg({'value': ['mean', 'std']})

    df.columns = ['metric', 'bidder', 'mean', 'std']

    df = df.loc[df['metric'].isin(metrics)]

    def map_mean_std(row):
        result = str(form.format(round(row['mean'], precision)))
        if with_stddev:
            result += ' (' + str(form.format(round(row['std'], precision))) \
                + ')'
        if result == 'nan (nan)':
            result = '--'
        return result
    df['value'] = df.apply(map_mean_std, axis=1)
    del df['mean'], df['std']
    df.set_index(['bidder', 'metric'], inplace=True)
    df = df.unstack(level='metric')
    df.columns = [y for (x, y) in df.columns]

    # establish requested column order
    df = df[metrics] 

    # bidder names
    if bidder_names is None:
        bidder_names = df.index
    df.insert(0, 'bidder', bidder_names)

    aliases = ALIASES_LATEX.copy()
    for m in metrics:
        if m[-5:-1] == '_bne':
            aliases[m] = aliases[m[:-5]][:-1] + '^\text{BNE{'+str(m[-1])+'}}$'

    df.columns = df.columns.map(
        lambda m: aliases[m] if m in aliases.keys() else m
    )

    return df


def df_to_tex(
        df: pd.DataFrame,
        name: str = 'table.tex',
        label: str = 'tab:full_reults',
        caption: str = '',
        index_names: bool = False,
    ):
    """Creates a tex file with the csv at `path` as a LaTeX table."""
    def bold(x):
        return r'\textbf{' + x + '}'
    df.to_latex(name, na_rep='--', escape=False,
                index=index_names, index_names=index_names, caption=caption,
                column_format='l'+'r'*(len(df.columns)-1), label=label,
                bold_rows=index_names, formatters={'bidder': bold})


def csv_to_boxplot(
        experiments: dict,
        name: str = 'boxplot.png',
        caption: str = 'caption',
        metric: str = 'eval/epsilon_relative',
        precision: int = 4
    ):
    """Creates a boxplot."""

    form = '{:.' + str(precision) + 'f}'

    aggregate_df = pd.DataFrame(columns=['gamma', 'locals', 'global'])
    for exp_name, exp_path in experiments.items():
        df = pd.read_csv(exp_path)
        end_epoch = df.epoch.max()
        df = df[df.epoch == end_epoch]
        df = df[df['tag'] == metric]

        single_df = pd.DataFrame(columns=['locals', 'global'])
        locals_ = df[df['subrun'] == 'locals'].value
        global_ = df[df['subrun'] == 'global'].value
        single_df['locals'] = locals_.to_numpy()
        single_df['global'] = global_.to_numpy()
        single_df['gamma'] = exp_name[-4:-1]
        aggregate_df = pd.concat([aggregate_df, single_df])

    # write to file
    c1 = '#1f77b4'
    c2 = '#ff7f0e'

    def setBoxColors(bp):
        plt.setp(bp['boxes'][0], color=c1)
        plt.setp(bp['caps'][0], color=c1)
        plt.setp(bp['caps'][1], color=c1)
        plt.setp(bp['whiskers'][0], color=c1)
        plt.setp(bp['whiskers'][1], color=c1)
        plt.setp(bp['fliers'][0], marker='.', markeredgecolor=c1)
        plt.setp(bp['medians'][0], color=c1)

        plt.setp(bp['boxes'][1], color=c2)
        plt.setp(bp['caps'][2], color=c2)
        plt.setp(bp['caps'][3], color=c2)
        plt.setp(bp['whiskers'][2], color=c2)
        plt.setp(bp['whiskers'][3], color=c2)
        plt.setp(bp['fliers'][1], marker='.', markeredgecolor=c2)
        plt.setp(bp['medians'][1], color=c2)

    fig = plt.figure(figsize=(4, 3))
    ax = plt.axes()
    pos = [1, 2]
    for gamma, _ in experiments.items():
        bp = plt.boxplot(aggregate_df[aggregate_df['gamma'] == gamma[-4:-1]] \
                         [['locals', 'global']].to_numpy(),
                         positions=pos, widths=1.5)
        setBoxColors(bp)
        pos = [p + 3 for p in pos]
    hB, = plt.plot([1, 1], color=c1)
    hR, = plt.plot([1, 1], color=c2)
    plt.legend((hB, hR), ('locals', 'global'), loc='lower right')
    ax.set_xticks(
        [1 + 31*float(gamma[-4:-1]) for gamma, _ in experiments.items()])
    ax.set_xticklabels(
        [float(gamma[-4:-1]) for gamma, _ in experiments.items()])
    # plt.xlim([0, 30])
    plt.ylim([-0.0015, 0.0015])
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.xlabel('correlation $\gamma$')
    plt.ylabel('loss ' + ALIASES_LATEX[metric])
    # plt.grid()
    plt.tight_layout()
    plt.savefig('experiments/' + name)


def bids_to_csv(
        experiments: dict,
        n_points: int = 1000
    ):
    """Load model, and save valuations and according actions from the model."""
    valuation = np.linspace(0, 2, n_points)

    return_dict = {'valuation': valuation}
    for _, exp_path in experiments.items():
        for model_path in os.listdir(exp_path + '/models'):
            model = NeuralNetStrategy.load(
                exp_path + '/models/' + model_path,
                device='cuda:1'
            )
            action = model.play(
                torch.tensor(valuation, dtype=torch.float).view(-1, 1)
            ).detach().numpy()
            action_dict = {
                f'action_{model_path[6:-3]}_{i}': action[:, i] for i in range(action.shape[-1])
            }
            return_dict = {**return_dict, **action_dict}
        df = pd.DataFrame(return_dict)
        df.to_csv(exp_path + '/actions.csv', index=False)


def multi_run_plot(path: str, metrics: list = ['market/utilities',
                   'eval/estimated_relative_ex_ante_util_loss'],
                   varied_param="['learning']['learner_hyperparams']['population_size']",
                   name='llllgg_analysis_batchsize', title=None, max_iter: int=None,
                   markevery=[50, 1], labels='agent_names', std_or_minmax: bool=True):
    """Create side-by-side plots that display the learning for multiple
    configurations of the same (possibly asymmetric) auction. Only tested for
    LLLLGG setting.

    TODO: Some parts like the legend are hard coded.
    """
    aggregate_logs = []
    configs = []
    for root, _, files in os.walk(path):
        for file in files:
            if str(file).endswith('full_results.csv'):
                aggregate_logs.append(root + '/' + file)
            if str(file).endswith('experiment_configurations.json'):
                configs.append(root + '/' + file)           

    # sort
    aggregate_logs = natural_sort(aggregate_logs)  # for llllgg_util(loss) plot change to: `[natural_sort(aggregate_logs)[0]]`
    configs = natural_sort(configs)  # for llllgg_util(loss) plot change to: `[natural_sort(configs)[0]]`
    # aggregate_logs = [natural_sort(aggregate_logs)[0]]  # for llllgg_util(loss) plot change to: `[natural_sort(aggregate_logs)[0]]`
    # configs = [natural_sort(configs)[0]]  # for llllgg_util(loss) plot change to: `[natural_sort(configs)[0]]`

    fig, axs = plt.subplots(nrows=1, ncols=len(metrics), figsize=(10, 4))
    for aggregate_log_path, config_path, c, m in zip(aggregate_logs, configs, colors, markers):
        try:
            df = pd.read_csv(aggregate_log_path)
            with open(config_path) as json_file:
                config = json.load(json_file)
            label = eval(f'config{varied_param}')

            t_mean = df[df['tag'] == 'time_per_step'].value.mean()
            t_std = df[df['tag'] == 'time_per_step'].value.std()
            print(f"Average runtime for {varied_param} = {label} is {round(t_mean, 4)} ({round(t_std, 4)}).")

            df = df.loc[df['tag'].isin(metrics)]
            if max_iter is not None:
                df = df[df.epoch <= max_iter]

            df = df.groupby(['subrun', 'epoch', 'tag'], as_index=False) \
                .agg({'value': ['mean', 'std', 'min', 'max']})
            df.columns = ['subrun', 'epoch', 'tag', 'mean', 'std', 'min', 'max']

            for i, (metric, ax) in enumerate(zip(metrics, axs)):
                for j, agent in enumerate(reversed(df['subrun'].unique())):
                    temp_df = df[df['tag'] == metric]
                    temp_df = temp_df[temp_df['subrun'] == agent]
                    temp_df = temp_df.dropna()
                    epoch = temp_df.epoch.to_numpy()
                    mean = temp_df['mean'].to_numpy()
                    if labels == 'agent_names':
                        run_label = f'{agent}: {label}'
                    elif labels == 'n_items':
                        run_label = f'$m = {label}$'
                    elif labels == 'agent_names_only':
                        run_label = f'{agent}'
                        c = colors[j]
                    ax.plot(epoch, mean, '-' if agent=='locals' else '--',
                            marker=m, markevery=markevery[0] if i==0 else markevery[1],
                            label=run_label, color=c)
                    if std_or_minmax:
                        std = temp_df['std'].to_numpy()
                        y_low, y_max = mean - std, mean + std
                    else:
                        y_low, y_max = temp_df['min'].to_numpy(), temp_df['max'].to_numpy()
                    ax.fill_between(epoch, y_low, y_max, alpha=.3, color=c)
                ax.set_xlabel('epoch')
                ax.set_ylabel(ALIASES_LATEX[metric] if metric in ALIASES_LATEX.keys() else metric)
                if metric == 'eval/estimated_relative_ex_ante_util_loss':
                    ax.set_yscale("log")
                    ax.set_ylim([.8e-2, 1.2])
                ax.grid(visible=True, linestyle='--')
                # ax.set_xlim(0, max(epoch))
        except Exception as e:
            print(e)
    axs[1].legend(title=title)
    plt.tight_layout()
    plt.savefig(f'{name}.pdf')
    return


def natural_sort(l: list): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def create_full_results_from_tb(path: str):
    """Takes TB logs from one experiment (with possibly multiple runs and
    different bidder types) and creates the `full_results.csv` if that was not
    created during execution.
    """
    runs = list(next(os.walk(path))[1])
    agents = ['globals', 'locals']

    full_logs = pd.DataFrame()
    for run in runs:
        for agent in agents:
            run_dir = path + '/' + run + '/' + agent
            run_dir += '/' + list(os.walk(run_dir))[0][2][0]

            event_acc = EventAccumulator(run_dir)
            event_acc.Reload()
            dframes = {}
            mnames = event_acc.Tags()['scalars']
            for n in mnames:
                dframes[n] = pd.DataFrame(event_acc.Scalars(n), columns=["wall_time", "epoch", n])
                dframes[n] = dframes[n].set_index("epoch")
            df = pd.concat([v for k, v in dframes.items()], axis=1)
            df['run'] = run
            df['subrun'] = agent
            df['epoch'] = df.index

            full_logs = pd.concat([full_logs, df], axis=0)

    full_logs = full_logs.melt(
        id_vars=['run', 'subrun', 'epoch'],
        value_vars=[
            'market/utilities',
            'learner_info/update_norm', 'learner_info/gradient_norm',
            'eval/util_loss_ex_ante', 'eval/util_loss_ex_interim', 
            'eval/estimated_relative_ex_ante_util_loss'
        ],
        var_name='tag'
    )

    full_logs.to_csv(path + '/full_results.csv')


if __name__ == '__main__':
    pass
