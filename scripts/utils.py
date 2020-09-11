"""utilities for run scripts"""
import pandas as pd

def csv_to_tex(
        data_path: str = '/home/kohring/bnelearn/experiments/single_item/first_price/interdependent/' \
            + 'uniform/symmetric/risk_neutral/2p/2020-09-11 Fri 14.21/aggregate_log.csv',
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
        'eval/util_loss_ex_interim': '$\hat \epsilon_$'
    }

    df = pd.read_csv(data_path)
    end_epoch = df.epoch.max()
    df = df[df.epoch == end_epoch]

    aggregate = df.groupby(['tag'], as_index=False).agg({'value': ['mean','std']})
    aggregate.columns = ['metric', 'mean','std']
    aggregate = aggregate.loc[aggregate['metric'].isin(
        ['eval/L_2', 'eval/epsilon_relative', 'eval/util_loss_ex_interim'])]
    aggregate.metric = aggregate.metric.map(lambda m: ALIASES[m])

    # write to file
    aggregate.to_latex(name, float_format="%.4f", escape=False, index=False, caption=caption)

#csv_to_tex()
