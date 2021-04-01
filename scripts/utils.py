"""utilities for run scripts"""
import os
import pandas as pd
import matplotlib.pyplot as plt


def logs_to_df(path: str):
    """Creates and returns a Pandas DataFrame from all logs in `path`"""
    experiments = [os.path.join(dp, f) for dp, dn, filenames
                   in os.walk(path) for f in filenames
                   if os.path.splitext(f)[1] == '.csv']
    experiments = {str(e): e for e in experiments}

    aggregate_df = pd.DataFrame()
    for exp_name, exp_path in experiments.items():
        if 'full_results' in exp_name:
            continue
        df = pd.read_csv(exp_path)
        df['exp'] = exp_name
        aggregate_df = pd.concat([aggregate_df, df])

    # write to file
    aggregate_df.to_csv('experiments/summary.csv', index=False)


def cycle_logs_to_csv(path: str = '../experiments/summary.csv'):
    df = pd.read_csv(path)
    del df['epoch'], df['run']

    def reduce_exp(row):
        return row['exp'][54:]
    df['exp'] = df.apply(reduce_exp, axis=1)

    def get_learner(v):
        s = 0
        e = v.find('Learner') + len('Learner')
        return v[s:e]
    df['learner'] = df['exp'].apply(get_learner)

    def get_game(v):
        if 'cycle_auction_v1' in v:
            return 'v1'
        elif 'cycle_auction_v2' in v:
            return 'v2'
        elif 'cycle_auction_v3' in v:
            return 'v3'
        elif 'cycle_auction_v4' in v:
            return 'v4'
    df['game'] = df['exp'].apply(get_game)

    del df['exp']
    df.to_csv('oa.csv')


def plot_cycle_game(path: str):
    experiments = [os.path.join(dp, f) for dp, dn, filenames
                   in os.walk(path) for f in filenames
                   if os.path.splitext(f)[1] == '.csv']
    experiments = {str(e): e for e in experiments}

    for exp_name, exp_path in experiments.items():
        if 'full_results' in exp_name:
            df = pd.read_csv(exp_path)
            df = df[df['tag'] == 'eval/actions']
            df = df[['subrun', 'epoch', 'value']]
            x = df[df['subrun'] == 'bidder0']['value'].to_numpy()
            y = df[df['subrun'] == 'bidder1']['value'].to_numpy()

            fig, ax = plt.subplots()
            ax.plot(x, y, label='learning')
            ax.plot(x[0], y[0], 'x', label='start point')
            ax.set_xlabel('action agent 1'); ax.set_ylabel('action agent 2')
            ax.plot(x[-1], y[-1], 'x', label='end point')
            ax.plot(0, 0, '.', label='origin', color='black')
            ax.set_box_aspect(1)
            plt.title(exp_name)
            plt.legend(loc='upper left')
            # plt.xlim([-1, 1])
            # plt.ylim([-1, 1])
            plt.savefig(exp_path + 'learning.png')


if __name__ == '__main__':

    path = '/home/kohring/bnelearn/experiments/opponent-awareness-default-hps'
    
    # extract log
    # logs_to_df(path=path)
    # cycle_logs_to_csv()

    # plot actions
    plot_cycle_game(path=path)
