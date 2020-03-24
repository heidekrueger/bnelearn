import os, sys, time
root_path = os.path.join(os.path.expanduser('~'), 'bnelearn')
if root_path not in sys.path:
    sys.path.append(root_path)

import scripts_work_in_progress.run_matrix_NN as NN
import scripts_work_in_progress.run_matrix_FP as FP
import random

games = ['MP']#['PD', 'MP', 'BoS', 'RPS','JG']#['PD', 'MP', 'BoS']
learners = ['NSP']#[ 'FP', 'FPS', 'FPM', 'NSP']

# smallest tau, update interval, tau (update size)
epochs = 10000
params_fp = [[0.2, 10, 0.9]]#[0.0001, 10, 0.99],[0.2, 10, 0.9],[0.5, 10,0.9]]
# batch size, learning_rate, lr_decay, lr_decay_every, lr_decay_factor, siga, n_perturbations
params_nsp = [[2**10, 1/5, False, 100, 0.9, 5, 10]]


equilibria = {'PD': [[[0,1],[1,0]]],
              'MP': [[[0.5,0.5],[0.5,0.5]]],
              'BoS': [
                        [[0.6,0.4],[0.4,0.6]],
                        [[1,0],[1,0]],
                        [[0,1],[0,1]]],
              'RPS': [
                        [[1/3,1/3,1/3],
                        [1/3,1/3,1/3]]],
              'JG': [[
                        [0.5,0.5],
                        [0.5,0.5],
                        [0.5,0.5]]]}

beliefs_tracking = [None] * len(games)
iterationen = 10

random.seed(1)

for g,game in enumerate(games):
    #create random beliefs and seeds for game 
    random_beliefs = [None] * iterationen
    seeds = [None] * iterationen
    for i in range(iterationen):
        n_players = len(equilibria[game][0])
        belief = [None] * n_players
        for player in range(n_players):
            actions = len(equilibria[game][0][player])
            belief[player] = [None] * actions
            for action in range(actions):
                belief[player][action] = random.random()
        random_beliefs[i] = belief
        seeds[i] = random.randint(1,1024)

    beliefs_tracking[g] = random_beliefs
    for learner in learners:
        if learner == 'NSP':
            params = params_nsp
            model = NN
        else:
            params = params_fp
            model = FP
        for param in params:
            for i, belief in enumerate(random_beliefs):
                # game, learner, param, belief, epoch
                # Skip easy first 2 iterations
                epochs_tmp = epochs
                #if i<2:
                #    epochs_tmp = 1000
                model.main([game, learner, param, belief, epochs_tmp, root_path, seeds[i]])