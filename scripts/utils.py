"""utilities for run scripts"""
import pandas as pd


def csv_to_tex(
        experiments: dict,
        name: str = 'table.tex',
        caption: str = 'caption'
    ):
    """Creates a tex file with the csv at `path` as a LaTeX table."""

    #pylint: disable=anomalous-backslash-in-string
    ALIASES = {
        'eval/L_2':                  '$L_2$',
        'eval/L_inf':                '$L_\infty$',
        'eval/epsilon_absolute':     '$\epsilon_\text{abs}$',
        'eval/epsilon_relative':     '$\mathcal L$',
        'eval/overhead_hours':       '$T$',
        'eval/update_norm':          '$|\Delta \theta|$',
        'eval/utilities':            '$u$',
        'eval/utility_vs_bne':       '$\hat u(\beta_i, \beta^*_{-i})$',
        'eval/util_loss_ex_ante':    '$\hat \ell$',
        'eval/util_loss_ex_interim': '$\hat \epsilon$',
        'eval/estimated_relative_ex_ante_util_loss': '$\hat \mathcal L$',
    }

    metrics = ['eval/L_2', 'eval/epsilon_relative', 'eval/util_loss_ex_interim',
               'eval/estimated_relative_ex_ante_util_loss']
    aggregate_df = pd.DataFrame(columns=metrics)
    for exp_name, exp_path in experiments.items():
        df = pd.read_csv(exp_path)
        end_epoch = df.epoch.max()
        df = df[df.epoch == end_epoch]

        single_df = df.groupby(['tag'], as_index=False).agg({'value': ['mean','std']})
        single_df.columns = ['metric', 'mean','std']
        single_df = single_df.loc[single_df['metric'].isin(metrics)]

        single_df[exp_name] = single_df.apply(
            lambda x: str('{:.4f}'.format(round(x['mean'], 4))) + ' (' \
                + str('{:.2f}'.format(round(x['std'], 2))) + ')',
            axis=1
        )
        single_df.index = single_df['metric']
        del single_df['mean'], single_df['std'], single_df['metric']
        aggregate_df = pd.concat([aggregate_df, single_df.T])

    aggregate_df.columns = aggregate_df.columns.map(lambda m: ALIASES[m])

    # write to file
    aggregate_df.to_latex('experiments/' + name, float_format="%.4f", escape=False, index=True, caption=caption)



if __name__ == '__main__':
    exps = {
        'Affiliated values': '/home/kohring/bnelearn/experiments/single_item/first_price/' \
            + 'interdependent/uniform/symmetric/risk_neutral/2p/2020-09-11 Fri 17.29/aggregate_log.csv',
        'Correlated values': '/home/kohring/bnelearn/experiments/single_item/second_price/' \
            + 'interdependent/uniform/symmetric/risk_neutral/3p/2020-09-11 Fri 17.29/aggregate_log.csv',
        'LLG ($\gamma=0$)': '/home/kohring/bnelearn/experiments/LLG/nearest_zero/independent/' \
            'gamma_0.0/2020-09-11 Fri 17.29/aggregate_log.csv'
    }

    csv_to_tex(
        experiments = exps,
        name = 'interdependent_table.tex',
        caption = 'Mean and standard deviation of experiments over ten runs each.'
    )
