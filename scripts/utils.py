"""utilities for run scripts"""
import os, sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'bnelearn'))

from bnelearn.strategy import NeuralNetStrategy
from bnelearn.experiment.configuration_manager import ConfigurationManager
from bnelearn.util import logging


#pylint: disable=anomalous-backslash-in-string
ALIASES = {
    'eval/L_2':                  '$L_2$',
    'eval/L_inf':                '$L_\infty$',
    'eval/epsilon_absolute':     '$\epsilon_\text{abs}$',
    'eval/epsilon_relative':     '$\mathcal{L}$',
    'eval/overhead_hours':       '$T$',
    'eval/update_norm':          '$|\Delta \theta|$',
    'eval/utilities':            '$u$',
    'eval/utility_vs_bne':       '$\hat u(\beta_i, \beta^*_{-i})$',
    'eval/util_loss_ex_ante':    '$\hat \ell$',
    'eval/util_loss_ex_interim': '$\hat \epsilon$',
    'eval/estimated_relative_ex_ante_util_loss': '$\hat{\mathcal{L}}$',
    'eval/efficiency':           '$\mathcal{E}$',
    'eval/revenue':              '$\mathcal{R}$'
}

def logs_to_df(
        path: str or dict,
        metrics: list = ['eval/epsilon_relative', 'eval/util_loss_ex_interim',
                         'eval/estimated_relative_ex_ante_util_loss',
                         'eval/efficiency', 'eval/utilities'],
        precision: int = 4,
        with_stddev: bool = False,
        with_setting_parameters: bool = True,
    ):
    """Creates and returns a Pandas DataFrame from all logs in `path`."""

    if isinstance(path, str):
        experiments = [os.path.join(dp, f) for dp, dn, filenames
                       in os.walk(path) for f in filenames
                       if os.path.splitext(f)[1] == '.csv']
        experiments = {str(e): e for e in experiments}
    else:
        experiments = path

    form = '{:.' + str(precision) + 'f}'

    columns = ['Auction game'] + metrics
    aggregate_df = pd.DataFrame(columns=columns)
    for exp_name, exp_path in experiments.items():
        df = pd.read_csv(exp_path)
        end_epoch = df.epoch.max()
        df = df[df.epoch == end_epoch]

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
        aggregate_df = pd.concat([aggregate_df, single_df.T])
        aggregate_df['Auction game'][-1] = exp_name

    aggregate_df.columns = aggregate_df.columns.map(
        lambda m: ALIASES[m] if m in ALIASES.keys() else m
    )

    if with_setting_parameters:
        def map_type(row):
            for t in ['Bernoulli', 'constant', 'independent']:
                if t in row['Auction game']:
                    return t
            return None
        aggregate_df['Corr Type'] = aggregate_df.apply(map_type, axis=1)

        def map_strength(row):
            if row['Corr Type'] == 'independent':
                return 0.0
            elif 'gamma_' in row['Auction game']:
                start = row['Auction game'].find('gamma_') + 6
                end = row['Auction game'].find('/', start)
                return row['Auction game'][start:end]
            return None
        aggregate_df['Corr Strength'] = pd.to_numeric(
            aggregate_df.apply(map_strength, axis=1))

        def map_risk(row):
            if 'risk_' in row['Auction game']:
                start = row['Auction game'].find('risk_') + 5
                end = row['Auction game'].find('/', start)
                return row['Auction game'][start:end]
            return 1.0
        aggregate_df['Risk'] = pd.to_numeric(
            aggregate_df.apply(map_risk, axis=1))

    # write to file
    aggregate_df.to_csv('experiments/summary.csv', index=False)

    return aggregate_df


def csv_to_tex(
        experiments: dict,
        name: str = 'table.tex',
        caption: str = 'caption',
        metrics: list = ['eval/L_2', 'eval/epsilon_relative',
                         'eval/estimated_relative_ex_ante_util_loss'],
        precision: int = 2,
    ):
    """Creates a tex file with the csv at `path` as a LaTeX table."""

    aggregate_df = logs_to_df(path=experiments, metrics=metrics,
                              precision=precision, with_stddev=True,
                              with_setting_parameters=False)

    # write to file
    aggregate_df.to_latex('experiments/' + name, float_format="%.4f",
                          na_rep='--', escape=False, index=False,
                          caption=caption, column_format='l'+'r'*len(metrics),
                          label='tab:full_results')


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


def plot_bid_functions(experiments: dict):
    """Plot LLG locals"""
    valuation_t = torch.linspace(0, 1, 1000).view(-1, 1)
    valuation_n = valuation_t.detach().numpy()

    with plt.style.context('grayscale'):
        plt.figure(figsize=(5, 4))
        plt.plot(valuation_n, valuation_n, label='truthful',
                 color='lightgrey', linestyle='dotted')
        for exp_name, exp_path in experiments.items():
            model = NeuralNetStrategy.load(
                exp_path + '/models/model_0.pt',
                device='cuda:1'
            )
            bid = model.play(valuation_t).detach().numpy()
            plt.plot(valuation_n, bid, label=exp_name)

        # plt.title('LLG local bidders')
        plt.legend()
        plt.xlabel('valuation'); plt.ylabel('bid')
        plt.xlim([0, 1]); plt.ylim([0, 1])
        plt.tight_layout()
        plt.savefig('experiments/interdependence/llg_bid_functions.eps')


if __name__ == '__main__':

    # logs_to_df(path='/home/kohring/bnelearn/experiments/comp_statics',
    #            precision=4)

    ### Create bid function plot ----------------------------------------------
    # exps = {
    #     '$\gamma = 0.1$': '/home/kohring/bnelearn/experiments/' + \
    #         'interdependence/risk-vs-correlation/LLG/nearest_vcg/' + \
    #             'Bernoulli_weights/gamma_0.1/risk_0.9/2020-10-26 Mon 13.58/00 09:43:03 0',
    #     '$\gamma = 0.5$': '/home/kohring/bnelearn/experiments/' + \
    #         'interdependence/risk-vs-correlation/LLG/nearest_vcg/' + \
    #         'Bernoulli_weights/gamma_0.5/risk_0.9/2020-10-26 Mon 13.58/00 10:41:46 0',
    #     '$\gamma = 0.9$': '/home/kohring/bnelearn/experiments/' + \
    #         'interdependence/risk-vs-correlation/LLG/nearest_vcg/' + \
    #         'Bernoulli_weights/gamma_0.9/risk_0.9/2020-10-26 Mon 13.58/00 11:40:35 0',
    # }
    # plot_bid_functions(exps)


    ### All experiments -------------------------------------------------------
    # exps = {
    #     'Affiliated values':  '/home/kohring/bnelearn/experiments/single_item/first_price/interdependent/uniform/symmetric/risk_neutral/2p/2020-09-18 Fri 20.53/aggregate_log.csv',
    #     'Cor. values':        '/home/kohring/bnelearn/experiments/single_item/second_price/interdependent/uniform/symmetric/risk_neutral/3p/2020-09-18 Fri 20.53/aggregate_log.csv',
    #     'Cor. values 10p':    '/home/kohring/bnelearn/experiments/single_item/second_price/interdependent/uniform/symmetric/risk_neutral/10p/2020-09-26 Sat 19.54/aggregate_log.csv',
    #     'LLG Bernoulli NZ':   '/home/kohring/bnelearn/experiments/LLG/nearest_zero/Bernoulli_weights/gamma_0.5/2020-10-02 Fri 20.59/aggregate_log.csv',
    #     'LLG Bernoulli VCG':  '/home/kohring/bnelearn/experiments/LLG/vcg/Bernoulli_weights/gamma_0.5/2020-10-02 Fri 20.59/aggregate_log.csv',
    #     'LLG Bernoulli NVCG': '/home/kohring/bnelearn/experiments/LLG/nearest_vcg/Bernoulli_weights/gamma_0.5/2020-10-02 Fri 20.59/aggregate_log.csv',
    #     'LLG Bernoulli NB':   '/home/kohring/bnelearn/experiments/LLG/nearest_bid/Bernoulli_weights/gamma_0.5/2020-10-02 Fri 20.59/aggregate_log.csv',
    #     'LLG constant NZ':    '/home/kohring/bnelearn/experiments/LLG/nearest_zero/constant_weights/gamma_0.5/2020-09-30 Wed 22.13/aggregate_log.csv',
    #     'LLG constant VCG':   '/home/kohring/bnelearn/experiments/LLG/vcg/constant_weights/gamma_0.5/2020-09-30 Wed 22.13/aggregate_log.csv',
    #     'LLG constant NVCG':  '/home/kohring/bnelearn/experiments/LLG/nearest_vcg/constant_weights/gamma_0.5/2020-09-30 Wed 22.13/aggregate_log.csv',
    #     'LLG constant NB':    '/home/kohring/bnelearn/experiments/LLG/nearest_bid/constant_weights/gamma_0.5/2020-09-30 Wed 22.13/aggregate_log.csv',
    # }

    # csv_to_tex(
    #     experiments = exps,
    #     name = 'interdependent_table.tex',
    #     caption = 'Mean and standard deviation of experiments over ten runs' \
    #         + ' each. For the LLG settings, a correlation of $\gamma = 0.5$' \
    #         + ' under risk-neutral bidders was chosen.'
    # )


    ### Comparison over differnt correlations ---------------------------------
    # TODO Nils: is broke
    # exp_time = '2020-10-02 Fri 21.00'
    # exps = {'$\gamma = 0.0$': '/home/kohring/bnelearn/experiments/LLG/nearest_zero/independent/' \
    #             + exp_time + '/aggregate_log.csv'}
    # for gamma in [g/10 for g in range(1, 11)]:
    #     exps.update({'$\gamma = {}$'.format(gamma):
    #         '/home/kohring/bnelearn/experiments/LLG/nearest_zero/Bernoulli_weights/' \
    #             + 'gamma_{}'.format(gamma) + '/' + exp_time + '/aggregate_log.csv'
    #     })

    # csv_to_boxplot(
    #     experiments = exps,
    #     metric = 'eval/epsilon_relative',
    #     name = 'boxplot.png',
    #     caption = 'Mean and standard deviation of experiments over four runs each.',
    #     precision = 4
    # )


    ### Risk vs correlation experiment ----------------------------------------
    path = '/home/kohring/bnelearn/experiments/interdependence/' + \
        'risk-vs-correlation-rne'
    metric = '$\mathcal{R}$' # either one of the metricies
    xaxis = 'Corr Strength' # either one in ['Corr Strength', 'Risk']

    df = logs_to_df(path=path)
    with plt.style.context('grayscale'):
        plt.figure(figsize=(5, 3))
        # plt.plot([0, 1], [0.01, 0.01], label='1%', color='lightgrey',
        #          linestyle='dotted')
        for corr_type in ['constant', 'Bernoulli']:
            df_sub = df[df['Corr Type'] == corr_type]
            corrs = sorted(pd.unique(df_sub['Corr Strength']))
            corrs.insert(0, 0)
            risks = sorted(pd.unique(df_sub['Risk']))
            data = np.zeros((len(corrs), len(risks)))
            for i, corr in enumerate(corrs):
                for j, risk in enumerate(risks):
                    if corr == 0:
                        data[i, j] = float(
                            df[df['Corr Strength'] == corr] \
                                [df['Risk'] == risk][metric][0]
                        )
                    else:
                        data[i, j] = float(
                            df_sub[df_sub['Corr Strength'] == corr] \
                                [df_sub['Risk'] == risk][metric]
                        )
            if xaxis == 'Corr Strength':
                x = corrs
                axis = 1
            else:
                x = risks
                axis = 0
            # plt.errorbar(x, data.mean(axis=axis),
            #              yerr=data.std(axis=axis),
            #              label=corr_type + ' weights', marker='o')
            plt.plot(x, data.mean(axis=axis), label=corr_type + ' weights')

            # fig, ax = plt.subplots()
            # plt.imshow(data, cmap='gray', interpolation='nearest')
            # plt.xlabel('risk parameter $\\rho$')
            # ax.set_xticklabels(corrs)
            # ax.set_xticks(np.arange(len(corrs)))
            # plt.ylabel('correlation strength $\gamma$')
            # ax.set_yticks(np.arange(len(risks)))
            # ax.set_yticklabels(risks)
            # plt.colorbar()


        if metric == '$u$':
            plt.ylim([0.2, 0.5])
        elif metric == '$\mathcal{E}$':
            plt.ylim([0.8, 1.0])
        elif metric == '$\hat{\mathcal{L}}$':
            plt.ylim([0.005, 0.015])

        if xaxis == 'Corr Strength':
            plt.xlim([0, 1])
            plt.xlabel('correlation strength $\gamma$')
        else:
            plt.xlim([0.1, 1.0])
            plt.xlabel('risk parameter $\\rho$')

        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()
        plt.savefig('experiments/interdependence/risk-vs-correlation/' + \
            '{}.eps'.format(metric))
