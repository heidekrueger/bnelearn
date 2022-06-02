"""Some helper functions for collecting and summarizing logging data.
"""
import os, sys
import torch
import pandas as pd
import numpy as np
import json

from bnelearn.strategy import NeuralNetStrategy
from bnelearn.util.metrics import ALIASES_LATEX


def multiple_exps_logs_to_df(
        path: str or dict,
        metrics: list = ['eval_vs_bne/L_2', 'eval/epsilon_relative', 'eval/util_loss_ex_interim',
                         'eval/estimated_relative_ex_ante_util_loss',
                         'eval/efficiency', 'eval/revenue', 'eval/utilities'],
        precision: int = 4,
        with_setting_parameters: bool = True,
        save: bool = False,
    ) -> pd.DataFrame:
    """Creates and returns a Pandas DataFrame from all logs in `path`.

    This function is universally usable.

    Arguments:
        path: str or dict, which path to crawl for csv logs.
        metrics: list of which metrics we want to load in the df.
        precision: int of how many decimals we request.
        with_stddev: bool.
        with_setting_parameters: bool, some hyperparams can be read in from
            the path itself. Turning this switch on pareses these values to
            individual columns.

    Returns:
        aggregate_df pandas dataframe with one run corresponding to one row,
            columns correspond to the logged metrics (from the last iter).

    """
    if isinstance(path, str):
        experiments = [os.path.join(dp, f) for dp, dn, filenames
                       in os.walk(path) for f in filenames
                       if os.path.splitext(f)[1] == '.csv'
                       and "aggregate_log" in f]
        experiments = {str(e): e for e in experiments}
    else:
        experiments = path

    if len(experiments) == 0:
        print("Path empty.")
        return pd.DataFrame()

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

        single_df[exp_name] = single_df.apply(
            lambda row: _map_mean_std(row, precision),
            axis=1
            )

        single_df.index = single_df['metric']
        del single_df['mean'], single_df['std'], single_df['metric']
        aggregate_df = pd.concat([aggregate_df, single_df.T])
        aggregate_df['Auction game'][-1] = exp_name

    aggregate_df.columns = aggregate_df.columns.map(
        lambda m: ALIASES_LATEX[m] if m in ALIASES_LATEX.keys() else m
    )

    if with_setting_parameters:
        add_config(aggregate_df)

    # write to file
    if save:
        aggregate_df.to_csv(f'{path}/summary.csv', index=False)

    return aggregate_df


def _map_mean_std(row, precision: int=4):
    """Combine mean and std into one string column."""
    form = '{:.' + str(precision) + 'f}'

    mean = str(form.format(round(row['mean'], precision)))
    std = ' (' + str(form.format(round(row['std'], precision))) + ')'
    result = mean + std
    return result if result != 'nan (nan)' else '--'


def _get_config(row, path_cut_off: str):
    """Read JSON configuration file."""
    with open(row['Auction game'][:-path_cut_off] + 'experiment_configurations.json') as json_file:
        experiment_config_as_dict = json.load(json_file)
    return experiment_config_as_dict


def add_config(aggregate_df, aggregate: bool = True):
    """Load config and add to dataframe."""
    
    path_cut_off = 17 if aggregate else 16

    def map_smoothing(row):
        experiment_config_as_dict = _get_config(row, path_cut_off)
        return experiment_config_as_dict["learning"]["smoothing_temperature"]
    smoothing_temperature = aggregate_df.apply(map_smoothing, axis=1)
    if smoothing_temperature.shape[0] > 0:
        aggregate_df['Smoothing'] = smoothing_temperature

    # multi-unit mappings
    def map_pricing(row):
        experiment_config_as_dict = _get_config(row, path_cut_off)
        payment_rule = experiment_config_as_dict["setting"]["payment_rule"]
        return ALIASES_LATEX[payment_rule]
    pri = aggregate_df.apply(map_pricing, axis=1)
    if pri.shape[0] > 0:
        aggregate_df['Pricing'] = pri

    def map_units(row):
        experiment_config_as_dict = _get_config(row, path_cut_off)
        return experiment_config_as_dict["setting"]["n_items"]
    uni = aggregate_df.apply(map_units, axis=1)
    if uni.shape[0] > 0:
        aggregate_df['Items'] = pd.to_numeric(uni)

    def map_players(row):
        experiment_config_as_dict = _get_config(row, path_cut_off)
        return experiment_config_as_dict["setting"]["n_players"]
    pla = aggregate_df.apply(map_players, axis=1)
    if pla.shape[0] > 0:
        aggregate_df['Players'] = pd.to_numeric(pla)

    def map_batch(row):
        experiment_config_as_dict = _get_config(row, path_cut_off)
        return experiment_config_as_dict["learning"]["batch_size"]
    bat = aggregate_df.apply(map_batch, axis=1)
    if bat.shape[0] > 0:
        aggregate_df['Batch'] = pd.to_numeric(bat)

    def map_corrtype(row):
        experiment_config_as_dict = _get_config(row, path_cut_off)
        return experiment_config_as_dict["setting"]["correlation_types"]
    cor = aggregate_df.apply(map_corrtype, axis=1)
    if cor.shape[0] > 0:
        aggregate_df['Corr Type'] = cor

    def map_strength(row):
        experiment_config_as_dict = _get_config(row, path_cut_off)
        gamma = experiment_config_as_dict["setting"]["gamma"]
        return 0.0 if row['Corr Type'] == 'independent' else gamma
    stre = aggregate_df.apply(map_strength, axis=1)
    if stre.shape[0] > 0:
        aggregate_df['Corr Strength'] = pd.to_numeric(stre)

    def map_risk(row):
        experiment_config_as_dict = _get_config(row, path_cut_off)
        return experiment_config_as_dict["setting"]["risk"]
    ris = aggregate_df.apply(map_risk, axis=1)
    if ris.shape[0] > 0:
        aggregate_df['Risk'] = pd.to_numeric(ris)


def single_asym_exp_logs_to_df(
        exp_path: str,
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
    df = pd.read_csv(exp_path)
    end_epoch = df.epoch.max()
    df = df[df.epoch == end_epoch]

    df = df.groupby(
        ['tag', 'subrun'], as_index=False
    ).agg({'value': ['mean', 'std']})

    df.columns = ['metric', 'bidder', 'mean', 'std']

    df = df.loc[df['metric'].isin(metrics)]

    df['value'] = df.apply(
        lambda row: _map_mean_std(row, precision),
        axis=1
        )

    del df['mean'], df['std']
    df.set_index(['bidder', 'metric'], inplace=True)
    df = df.unstack(level='metric')
    df.columns = [y for (x, y) in df.columns]

    # bidder names
    if bidder_names is None:
        bidder_names = df.index
    df.insert(0, 'bidder', bidder_names)

    aliasies = ALIASES_LATEX.copy()
    for m in metrics:
        if m[-5:-1] == '_bne':
            aliasies[m] = aliasies[m[:-5]][:-1] + '^\text{BNE{'+str(m[-1])+'}}$'

    df.columns = df.columns.map(
        lambda m: aliasies[m] if m in aliasies.keys() else m
    )

    df["Auction game"] = exp_path
    add_config(df, aggregate=False)
    del df["Auction game"]

    return df


def df_to_tex(
        df: pd.DataFrame,
        name: str = 'table.tex',
        label: str = 'tab:full_reults',
        caption: str = '',
        save_path: str = None,
    ):
    """Creates a tex file with the csv at `path` as a LaTeX table."""
    def bold(x):
        return r'\textbf{' + x + '}'
    
    if save_path is None:
        save_path = os.path.dirname(os.path.realpath(__file__))

    df.to_latex(
        save_path + "/" + name, na_rep='--', escape=False,
        index=False, index_names=False, caption=caption,
        column_format='l'+'r'*(len(df.columns)-1),
        label=label,
        formatters={'bidder': bold}
        )


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


if __name__ == '__main__':
    pass
