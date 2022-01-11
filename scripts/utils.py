"""utilities for run scripts"""
import os, sys
import torch
import pandas as pd
import numpy as np
import json

import matplotlib.pyplot as plt
markers = ['o', '^', 's', 'p', '.', '+']
colors = [(0/255.,191/255.,196/255.), (248/255.,118/255.,109/255.),
          (150/255.,120/255.,170/255.), (255/255.,215/255.,130/255.)] 

sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'bnelearn'))

from bnelearn.strategy import NeuralNetStrategy
from bnelearn.experiment.configuration_manager import ConfigurationManager
from bnelearn.util import logging
from bnelearn.util.metrics import ALIASES_LATEX

#pylint: disable=anomalous-backslash-in-string
# TODO replace by: from bnelearn.util.metrics import ALIASES_LATEX as ALIASES
ALIASES = {
    'eval/epsilon':              '$\epsilon$',
    'eval_vs_bne/L_2':           '$L_2$',
    'eval_vs_bne/L_inf':         '$L_\infty$',
    'eval/epsilon_absolute':     '$\epsilon_\text{abs}$',
    'eval/epsilon_relative':     '$\mathcal{L}$',
    'eval/overhead_hours':       '$T$',
    'eval/update_norm':          '$|\Delta \theta|$',
    'market/utilities':          '$u$',
    'eval/utility_vs_bne':       '$\hat u(\beta_i, \beta^*_{-i})$',
    'eval/util_loss_ex_ante':    '$\hat \ell$',
    'eval/util_loss_ex_interim': '$\hat \epsilon$',
    'eval/estimated_relative_ex_ante_util_loss': '$\hat{\mathcal{L}}$',
    'eval/efficiency':           '$\mathcal{E}$',
    'eval/revenue':              '$\mathcal{R}$'
}

SETTING_ALIASES = {
    'correlation_types':         'Corr type',
    'correlation_coefficients':  'Corr strength',
    'risk':                      'risk $\rho$',
    'payment_rule':              'payment rule',
    'n_items':                   'items $m$',
    'n_players':                 'players $n$'
}


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
        with open(path + '/experiment_configurations.json') as json_file:
            config = json.load(json_file)
        for param in setting_parameters:
            single_df[param] = config['setting'][param]

        aggregate_df = pd.concat([aggregate_df, single_df])
        aggregate_df['Auction game'][-1] = exp_name

    aggregate_df.columns = aggregate_df.columns.map(
        lambda m: (ALIASES | SETTING_ALIASES)[m]
            if m in (ALIASES | SETTING_ALIASES).keys() else m)

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

    aliases = ALIASES.copy()
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
    ):
    """Creates a tex file with the csv at `path` as a LaTeX table."""
    def bold(x):
        return r'\textbf{' + x + '}'
    df.to_latex('experiments/' + name, na_rep='--', escape=False,
                index=False, index_names=False, caption=caption, column_format='l'+'r'*(len(df.columns)-1),
                label=label, formatters={'bidder': bold})


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
    plt.ylabel('loss ' + ALIASES[metric])
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
                   name='llllgg_analysis_batchsize'):
    """Create side-by-side plots that display the learning for multiple
    configurations of the same (possibly asymmetric) auction. Only tested for
    LLLLGG setting.

    TODO: Some parts like the legend are hard coded.
    """
    runs = [sub_path for sub_path in os.listdir(path)
            if os.path.isdir(os.path.join(path, sub_path))]
    aggregate_logs = []
    configs = []
    for run in runs:
        aggregate_logs.append(path+ '/' + run + '/full_results.csv')
        configs.append(path + '/' + run + '/experiment_configurations.json')

    fig, axs = plt.subplots(nrows=1, ncols=len(metrics), figsize=(10, 4))
    for aggregate_log_path, config_path, c, m in zip(aggregate_logs, configs, colors, markers):
        try:
            df = pd.read_csv(aggregate_log_path)
            with open(config_path) as json_file:
                config = json.load(json_file)
            label = eval(f'config{varied_param}')

            df = df.loc[df['tag'].isin(metrics)]
            df = df.groupby(['subrun', 'epoch', 'tag'], as_index=False) \
                .agg({'value': ['mean', 'std']})
            df.columns = ['subrun', 'epoch', 'tag', 'mean','std']

            for i, (metric, ax) in enumerate(zip(metrics, axs)):
                for agent in df['subrun'].unique():
                    temp_df = df[df['tag'] == metric][df['subrun'] == agent]
                    epoch = temp_df.epoch.to_numpy()
                    mean = temp_df['mean'].to_numpy()
                    std = temp_df['std'].to_numpy()
                    ax.plot(epoch, mean, '-' if agent=='locals' else '--',
                            marker=m, markevery=50 if i==0 else 1,
                            label=f'{agent}: {label}', color=c)
                    ax.fill_between(epoch, mean - std, mean + std, alpha=.3,
                                    color=c)
                ax.set_xlabel('epoch')
                ax.set_ylabel(ALIASES_LATEX[metric] if metric in ALIASES_LATEX.keys() else metric)
                if metric == 'eval/estimated_relative_ex_ante_util_loss':
                    ax.set_yscale("log")
                ax.grid(visible=True, linestyle='--')
                # ax.set_xlim(0, max(epoch))
        except Exception as e:
            pass
    axs[1].legend(
        title='bidder and corre-\nsponding batch size' if name=='llllgg_analysis_batchsize' \
              else 'bidder and corresponding\npopulation size', loc='upper right')
    plt.tight_layout()
    plt.savefig(f'{name}.pdf')
    return


if __name__ == '__main__':

    # Single item asymmetric uniform overlapping
    path = '/home/kohring/bnelearn/experiments/asymmmetric-final-results/single_item/first_price/uniform/asymmetric/risk_neutral/2p/2021-11-23 Tue 16.23/aggregate_log.csv'
    df = single_asym_exp_logs_to_df(
        path, metrics=['eval_vs_bne/L_2', 'eval/epsilon_relative',
        'eval/estimated_relative_ex_ante_util_loss', 'eval/util_loss_ex_interim'])
    df_to_tex(df, name='table_asym_overlapping.tex',
              caption='Average utilities achieved in asymmetric first-price setting with overlapping valuations. Mean and standard deviation are aggregated over ten runs of 2{,}000 iterations each.',
              label='table:asym_over_results')

    # Single-item asymmetric uniform disjunct
    path = '/home/kohring/bnelearn/experiments/asymmmetric-final-results/single_item/first_price/uniform/asymmetric/risk_neutral/2p/2021-11-23 Tue 22.26/aggregate_log.csv'
    df = single_asym_exp_logs_to_df(
        path, metrics=['eval_vs_bne/L_2_bne2', 'eval/epsilon_relative_bne2',
        'eval/estimated_relative_ex_ante_util_loss', 'eval/util_loss_ex_interim'])
    df_to_tex(df, name='table_asym_nonoverlapping.tex',
              caption='Average NPGA utilities achieved in asymmetric first-price setting with non-overlapping valuations. Aggregated over ten runs of 2{,}000 iterations each. Compared against the second equilibrium of \cite{kaplan2015multiple}.',
    	      label='table:asym_nonover_results')

    # # Single-item beta setting
    # Note: Not used in paper
    # path = '/home/kohring/bnelearn/experiments/asymmmetric-final-results/single_item/first_price/non-common/1.0risk/2players/2021-11-24 Wed 04.57/aggregate_log.csv'
    # df = single_asym_exp_logs_to_df(
    #     path, metrics=['eval/estimated_relative_ex_ante_util_loss', 'eval/util_loss_ex_interim'])
    # df_to_tex(df, name='table_asym_nonoverlapping.tex',
    #           caption='Average NPGA utilities achieved in single item setting with beta distributed prior. Aggregated over ten runs of 2{,}000 iterations each.',
    # 	        label='table:asym_beta')

    # Asymmetric LLG setting
    path = '/home/kohring/bnelearn/experiments/asymmmetric-final-results/LLGFull/mrcs_favored/independent/2021-11-24 Wed 11.39/aggregate_log.csv'
    df = single_asym_exp_logs_to_df(
        path, metrics=['eval_vs_bne/L_2', 'eval/epsilon_relative',
        'eval/estimated_relative_ex_ante_util_loss', 'eval/util_loss_ex_interim'])
    df_to_tex(df, name='table_llgfull.tex',
              caption='Results in the asymmetric LLG setting after 2{,}000 iterations and averaged over ten repetitions. Shown are the mean and standard deviation.',
    	      label='table:asym_nonover_results')

    # LLLLGG setting
    path = '/home/kohring/bnelearn/experiments/asymmmetric-final-results/LLLLGG/nearest_vcg/6p/2021-12-01 Wed 10.07'
    df = multiple_exps_logs_to_df(
        path, metrics=['eval/estimated_relative_ex_ante_util_loss',
        'eval/util_loss_ex_interim'], with_stddev=True)
    cols = ['payment rule', '$\hat{\mathcal{L}}$', '$\hat \epsilon$']
    df_to_tex(df[cols], name='table_llllgg.tex', label='table:auction-results-llllgg',
        caption='Results of NPGA after 5{,}000 (1{,}000) iterations in the LLLLGG first-price (nearest-vcg) auction. Results are averages over 10 (2) replications and the standard deviation displayed in brackets.',)

    # Scalability experiments
    path = '/home/kohring/bnelearn/experiments/asymmmetric-debug/varied-population_size/LLLLGG/first_price/6p'
    multi_run_plot(path, varied_param="['learning']['learner_hyperparams']['population_size']",
                   name='llllgg_analysis_popsize')
    path = '/home/kohring/bnelearn/experiments/asymmmetric-debug/varied-batch_size/LLLLGG/first_price/6p'
    multi_run_plot(path, varied_param="['learning']['batch_size']",
                   name='llllgg_analysis_batchsize')
