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


def cycle_logs_to_csv(path: str = 'experiments/summary.csv'):
    df = pd.read_csv(path)
    del df['epoch'], df['run']

    def get_learner(v):
        word = 'Learner'
        l = len(word)
        e = v.find(word) + l
        s = e - l
        while v[s] != '/':
            s = s-1
        return v[s+1:e]
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

    def get_game_radius(v):
        s = v.find('_r') + 2
        e = s + 1
        while v[e] != '/':
            e += 1
        return v[s:e]
    df['radius'] = df['exp'].apply(get_game_radius)

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
            runs = df.run.unique()
            for run in runs:
                df_run = df[df['run'] == run]
                df_run = df_run[['subrun', 'epoch', 'value']]
                x = df_run[df_run['subrun'] == 'bidder0']['value'].to_numpy()
                y = df_run[df_run['subrun'] == 'bidder1']['value'].to_numpy()

                _, ax = plt.subplots()
                ax.plot(x, y, label='learning')
                ax.plot(x[0], y[0], 'x', label='start point')
                ax.set_xlabel('action agent 1')
                ax.set_ylabel('action agent 2')
                ax.plot(x[-1], y[-1], 'x', label='end point')
                ax.plot(0, 0, '.', label='origin', color='black')
                ax.set_box_aspect(1)
                plt.title(exp_name)
                plt.legend(loc='upper left')
                # plt.xlim([-1, 1])
                # plt.ylim([-1, 1])
                plt.savefig(exp_name + run + ' learning.png')
                plt.close()


if __name__ == '__main__':

    path = '/home/kohring/bnelearn/experiments/opponent-awareness-specific-v4'

    # Extract log
    # logs_to_df(path=path)
    # cycle_logs_to_csv()

    # Plot actions
    plot_cycle_game(path=path)
