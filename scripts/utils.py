"""utilities for run scripts"""
import pandas as pd
import matplotlib.pyplot as plt

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
}


def csv_to_tex(
        experiments: dict,
        name: str = 'table.tex',
        caption: str = 'caption',
        metrics: list = ['eval/L_2', 'eval/epsilon_relative', 'eval/util_loss_ex_interim',
            'eval/estimated_relative_ex_ante_util_loss'],
        precision: int = 2,
    ):
    """Creates a tex file with the csv at `path` as a LaTeX table."""

    form = '{:.' + str(precision) + 'f}'

    columns = ['Auction game'] + metrics
    aggregate_df = pd.DataFrame(columns=columns)
    for exp_name, exp_path in experiments.items():
        df = pd.read_csv(exp_path)
        end_epoch = df.epoch.max()
        df = df[df.epoch == end_epoch]

        single_df = df.groupby(['tag'], as_index=False).agg({'value': ['mean','std']})
        single_df.columns = ['metric', 'mean','std']
        single_df = single_df.loc[single_df['metric'].isin(metrics)]

        single_df[exp_name] = single_df.apply(
            lambda x: str(form.format(round(x['mean'], precision))) + ' (' \
                + str(form.format(round(x['std'], precision))) + ')',
            axis=1
        )
        single_df.index = single_df['metric']
        del single_df['mean'], single_df['std'], single_df['metric']
        aggregate_df = pd.concat([aggregate_df, single_df.T])
        aggregate_df['Auction game'][-1] = exp_name

    aggregate_df.columns = aggregate_df.columns.map(
        lambda m: ALIASES[m] if m in ALIASES.keys() else m
    )

    # write to file
    aggregate_df.to_latex('experiments/' + name, float_format="%.4f", na_rep='--', escape=False,
                          index=False, caption=caption, column_format='l' + 'r'*len(metrics),
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
        bp = plt.boxplot(aggregate_df[aggregate_df['gamma'] == gamma[-4:-1]][['locals', 'global']].to_numpy(),
                         positions=pos, widths=1.5)
        setBoxColors(bp)
        pos = [p + 3 for p in pos]
    hB, = plt.plot([1, 1], color=c1)
    hR, = plt.plot([1, 1], color=c2)
    plt.legend((hB, hR), ('locals', 'global'), loc='lower right')
    ax.set_xticks([1 + 31*float(gamma[-4:-1]) for gamma, _ in experiments.items()])
    ax.set_xticklabels([float(gamma[-4:-1]) for gamma, _ in experiments.items()])
    # plt.xlim([0, 30])
    plt.ylim([-0.0015, 0.0015])
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.xlabel('correlation $\gamma$')
    plt.ylabel('loss ' + ALIASES[metric])
    # plt.grid()
    plt.tight_layout()
    plt.savefig('experiments/' + name)


if __name__ == '__main__':

    # All experiments
    exps = {
        'Affiliated values': '/home/kohring/bnelearn/experiments/single_item/first_price/interdependent/uniform/symmetric/risk_neutral/2p/2020-09-18 Fri 20.53/aggregate_log.csv',
        'Cor. values': '/home/kohring/bnelearn/experiments/single_item/second_price/interdependent/uniform/symmetric/risk_neutral/3p/2020-09-18 Fri 20.53/aggregate_log.csv',
        'LLG Bernoulli NZ': '/home/kohring/bnelearn/experiments/LLG/nearest_zero/Bernoulli_weights/gamma_0.5/2020-09-16 Wed 20.15/aggregate_log.csv',
        'LLG Bernoulli VCG': '/home/kohring/bnelearn/experiments/LLG/vcg/Bernoulli_weights/gamma_0.5/2020-09-28 Mon 11.04/aggregate_log.csv',
        'LLG Bernoulli P': '/home/kohring/bnelearn/experiments/LLG/proxy/Bernoulli_weights/gamma_0.5/2020-09-28 Mon 11.04/aggregate_log.csv',
        'LLG Bernoulli NVCG': '/home/kohring/bnelearn/experiments/LLG/nearest_vcg/Bernoulli_weights/gamma_0.5/2020-09-28 Mon 11.04/aggregate_log.csv',
        'LLG Bernoulli NB': '/home/kohring/bnelearn/experiments/LLG/nearest_bid/Bernoulli_weights/gamma_0.5/2020-09-28 Mon 11.04/aggregate_log.csv',
        'LLG constant': '/home/kohring/bnelearn/experiments/LLG/nearest_zero/constant_weights/gamma_0.5/2020-09-21 Mon 09.18/aggregate_log.csv',
        'Cor. values 10p': '/home/kohring/bnelearn/experiments/single_item/second_price/interdependent/uniform/symmetric/risk_neutral/10p/2020-09-26 Sat 19.54/aggregate_log.csv'
    }

    csv_to_tex(
        experiments = exps,
        name = 'interdependent_table.tex',
        caption = 'Mean and standard deviation of experiments over ten runs each. For the LLG settings, ' \
            + 'a correlation of $\gamma = 0.5$ was chosen.'
    )

    # # Comparison over differnt correlations
    # exp_time = '2020-09-16 Wed 20.15'
    # exps = {'$\gamma = 0.0$': '/home/kohring/bnelearn/experiments/LLG/nearest_zero/independent/' \
    #             + '/' + exp_time + '/aggregate_log.csv'}
    # for gamma in [g/10 for g in range(1, 11)]:
    #     exps.update({'$\gamma = {}$'.format(gamma):
    #         '/home/kohring/bnelearn/experiments/LLG/nearest_zero/Bernoulli_weights/' \
    #             + 'gamma_{}'.format(gamma) + '/' + exp_time + '/aggregate_log.csv'
    #     })

    # csv_to_boxplot(
    #     experiments = exps,
    #     metric = 'eval/epsilon_relative',
    #     name = 'boxplot.eps',
    #     caption = 'Mean and standard deviation of experiments over four runs each.',
    #     precision = 4
    # )
    