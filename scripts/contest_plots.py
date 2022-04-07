"""
Runs predefined experiments with individual parameters
fire.Fire() asks you to decide for one of the experiments defined above
by writing its name and define the required (and optional) parameters
e.g.:
    experiment.py single_item_uniform_symmetric 1 20 [2,3] 'first_price'

alternatively instead of fire.Fire() use, e.g.:
    single_item_uniform_symmetric(1,20,[2,3],'first_price')

"""
import os
import sys
from unittest import result
from numpy import number
os.CUDA_LAUNCH_BLOCKING=1

import torch
import multiprocessing as mp
import datetime

from itertools import product
from matplotlib import pyplot as plt

# put bnelearn imports after this.
# pylint: disable=wrong-import-position
sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'bnelearn'))

from bnelearn.experiment.configuration_manager import ConfigurationManager  # pylint: disable=import-error

# todo: train on multiple gpus -> improved runtime
# todo: output von npga ignorieren

if __name__ == '__main__':

    # path is user-specific
    log_root_dir = os.path.join(os.path.expanduser('~'), 'bnelearn', 'experiments', 'contest_plots', str(datetime.datetime.now()))

    def run_exp(use_valuation, impact_factors, corr_coefs, cost_types, cost_params, exp_type="tullock_contest", num_players=2, prios_lo=None, prior_hi=None, gpus=[5, 6]):

        results_loop = {}

        for impact_factor in impact_factors:

            # Todo: Find better solution for gpu blocking...
            if use_valuation:
                gpu = gpus[0]
            else:
                gpu = gpus[1]

            if corr_coefs is None:
                if cost_types is None:
                    if exp_type != 'crowdsourcing' and priors_lo is None:
                        # Contest Experiments
                        experiment_config, experiment_class = ConfigurationManager(experiment_type=exp_type, n_runs=1, n_epochs=3500) \
                            .set_setting(impact_factor=impact_factor) \
                            .set_logging(log_root_dir=log_root_dir, util_loss_frequency=1000, best_response=True) \
                            .set_hardware(specific_gpu=gpu) \
                            .set_learning(pretrain_iters=500, use_valuation=use_valuation, batch_size=2 ** 22) \
                            .get_config()

                        experiment = experiment_class(experiment_config)
                        _, models = experiment.run()
                        torch.cuda.empty_cache()

                        # set each model to eval mode
                        # tbd: assume no symmetry and model sharing -> only single model
                        model = models[0]
                        model.eval()

                        # store policy
                        results_loop[use_valuation, impact_factor, None, None, None, None, None, None, None] = model
                    elif exp_type != "crowdsourcing" and priors_lo is not None:
                        for i, _ in enumerate(priors_lo):
                            # Contest Experiments
                            experiment_config, experiment_class = ConfigurationManager(experiment_type=exp_type, n_runs=1, n_epochs=3500) \
                                .set_setting(impact_factor=impact_factor, u_lo=priors_lo[i], u_hi=priors_hi[i]) \
                                .set_logging(log_root_dir=log_root_dir, util_loss_frequency=1000, best_response=True) \
                                .set_hardware(specific_gpu=gpu) \
                                .set_learning(pretrain_iters=500, use_valuation=use_valuation, batch_size=2 ** 22) \
                                .get_config()

                            experiment = experiment_class(experiment_config)
                            _, models = experiment.run()
                            torch.cuda.empty_cache()

                            # set each model to eval mode
                            # tbd: assume no symmetry and model sharing -> only single model
                            model = models[0]
                            model.eval()

                            # store policy
                            if len(priors_lo[i]) > 1:
                                results_loop[use_valuation, impact_factor, None, None, None, None, None, priors_lo[i][1], priors_hi[i][1]] = model
                            else:
                                results_loop[use_valuation, impact_factor, None, None, None, None, None, priors_lo[i][0], priors_hi[i][0]] = model
                    else:
                        for np in num_players:
                            # define valuations
                            valuations = [0] * np
                            valuations[0] = impact_factor
                            valuations[1] = 1 - impact_factor
                            valuations = torch.tensor(valuations)

                            # Crowdsourcing Experiments
                            experiment_config, experiment_class = ConfigurationManager(experiment_type=exp_type, n_runs=1, n_epochs=7500) \
                                .set_setting(valuations=valuations, n_players=np) \
                                .set_logging(log_root_dir=log_root_dir, util_loss_frequency=1000) \
                                .set_hardware(specific_gpu=gpu) \
                                .set_learning(pretrain_iters=500, use_valuation=use_valuation, batch_size=2 ** 22) \
                                .get_config()

                            experiment = experiment_class(experiment_config)
                            _, models = experiment.run()
                            torch.cuda.empty_cache()

                            # set each model to eval mode
                            # tbd: assume no symmetry and model sharing -> only single model
                            model = models[0]
                            model.eval()

                            # store policy
                            results_loop[use_valuation, None, None, None, None, impact_factor, np, None, None] = model
                else:
                    for cost_type in cost_types:
                        for cost_param in cost_params:
                            # Contest Experiments
                            experiment_config, experiment_class = ConfigurationManager(experiment_type=exp_type, n_runs=1, n_epochs=3500)\
                                .set_setting(impact_factor=impact_factor, cost_type=cost_type, cost_param=cost_param) \
                                .set_logging(log_root_dir=log_root_dir, util_loss_frequency=1000) \
                                .set_hardware(specific_gpu=gpu) \
                                .set_learning(pretrain_iters=500, use_valuation=use_valuation, batch_size=2 ** 22) \
                                .get_config()

                            experiment = experiment_class(experiment_config)
                            _, models = experiment.run()
                            torch.cuda.empty_cache()

                            # set each model to eval mode
                            # tbd: assume no symmetry
                            model = models[0]
                            model.eval()

                            # store policy
                            results_loop[use_valuation, impact_factor, None, cost_type, cost_param, None, None, None, None] = model
            else:
                
                for corr in corr_coefs:

                    # Contest Experiments
                    experiment_config, experiment_class = ConfigurationManager(experiment_type=exp_type, n_runs=1, n_epochs=3500)\
                        .set_setting(impact_factor=impact_factor,  correlation_groups=[[0, 1]], correlation_coefficients=[corr], correlation_types='corr_type') \
                        .set_logging(log_root_dir=log_root_dir, util_loss_frequency=1000) \
                        .set_hardware(specific_gpu=gpu) \
                        .set_learning(pretrain_iters=500, use_valuation=use_valuation, batch_size=2 ** 22) \
                        .get_config()

                    experiment = experiment_class(experiment_config)
                    _, models = experiment.run()
                    torch.cuda.empty_cache()

                    # set each model to eval mode
                    # tbd: assume no symmetry
                    model = models[0]
                    model.eval()

                    # store policy
                    results_loop[use_valuation, impact_factor, corr, None, None, None, None, None, None] = model

        return results_loop

    def plot_config(config, exp_type, c_title, gpus):

        # unload config
        if len(config) == 2:
            use_valuations, impact_factors = config
            corr_coefs, cost_types, cost_params, num_players, priors_lo, priors_hi = [None] * 6
        elif len(config) == 3:
            use_valuations, impact_factors, corr_coefs = config 
            cost_types, cost_params, num_players, priors_lo, priors_hi = [None] * 5
        elif len(config) == 4 and config[3] != "crowdsourcing":
            use_valuations, impact_factors, cost_types, cost_params = config
            corr_coefs, num_players, priors_lo, priors_hi = [None] * 4
        elif len(config) == 4 and config[3] == "crowdsourcing":
            use_valuations, impact_factors, num_players, _ = config
            corr_coefs, cost_types, cost_params, priors_lo, priors_hi = [None] * 5
        elif len(config) == 5:
            use_valuations, impact_factors, priors_lo, priors_hi, _ = config
            corr_coefs, cost_types, cost_params, num_players = [None] * 4
        else:
            raise NotImplementedError

        # run async cost and valuations
        # pool = mp.Pool(mp.cpu_count())
        # results = [pool.apply_async(run_exp, args=(row, impact_factors, corr_coefs, cost_types, cost_params, exp_type, num_players, priors_lo, priors_hi, gpus)) for row in use_valuations]
        # pool.close()
        # pool.join()

        # results = [r.get() for r in results]

        results = [run_exp(use_valuations, impact_factors, corr_coefs, cost_types, cost_params, exp_type, num_players, priors_lo, priors_hi, gpus)]

        # plot policies
        for model_dict in results:

            use_valuations_result = list(model_dict.keys())[0][0]
            affiliatedModel = list(model_dict.keys())[0][2] is not None
            cost_adjusted = list(model_dict.keys())[0][3] is not None
            crowdsourcing = list(model_dict.keys())[0][5] is not None
            asymmetric = list(model_dict.keys())[0][7] is not None

            if use_valuations_result:
                gpu = gpus[0]
            else:
                gpu = gpus[1]

            cv = torch.linspace(0.1, 1.1, 100).to(F'cuda:{gpu}').unsqueeze(-1)

            if affiliatedModel:

                for impact_factor in impact_factors:
                    
                    plt.figure(figsize=(6,5))

                    for key in model_dict:
                        if key[1] != impact_factor:
                            continue
                        
                        pred = model_dict[key](cv).cpu().detach()
                        ifactor = key[1]
                        plt.plot(cv.cpu(), pred, marker='*', label=f'corr={key[2]}')

                    plt.legend()
                    plt.title(f'Numerical solution of {c_title} (m={impact_factor})', fontsize=14)
                    plt.ylabel('effort/bid', fontsize=13)
                    if use_valuations_result:
                        plt.xlabel('valuations', fontsize=13)
                        t = 'valuations'
                    else:
                        plt.xlabel('marginal cost', fontsize=13)
                        t = 'marginal_costs'

                    plt.savefig(f'NPGA_{c_title}_{t}_{impact_factor}.pdf')
            elif cost_adjusted:
                for impact_factor in impact_factors:
                    for cost_type in cost_types:
                    
                        plt.figure(figsize=(6,5))

                        for key in model_dict:
                            if key[1] != impact_factor or key[3] != cost_type:
                                continue
                            
                            pred = model_dict[key](cv).cpu().detach()
                            ifactor = key[1]
                            plt.plot(cv.cpu(), pred, marker='*', label=f'cost={key[4]}')

                        plt.legend()
                        plt.title(f'Numerical solution of {c_title} (m={impact_factor})', fontsize=14)
                        plt.ylabel('effort/bid', fontsize=13)
                        if use_valuations_result:
                            plt.xlabel('valuations', fontsize=13)
                            t = 'valuations'
                        else:
                            plt.xlabel('marginal cost', fontsize=13)
                            t = 'marginal_costs'

                        plt.savefig(f'NPGA_{c_title}_{t}_{impact_factor}.pdf')
            elif crowdsourcing:

                for np in num_players:

                    plt.figure(figsize=(6,5))

                    for key in model_dict:
                        if key[6] != np:
                            continue

                        pred = model_dict[key](cv).cpu().detach()
                        v = key[1]
                        plt.plot(cv.cpu(), pred, marker='*', label=f'V1={v}')

                    plt.legend()
                    plt.title(f'Numerical solution of {c_title.replace("x", str(np))}', fontsize=12)
                    plt.ylabel('effort/bid', fontsize=13)
                    if use_valuations_result:
                        plt.xlabel('valuations', fontsize=13)
                        t = 'valuations'
                    else:
                        plt.xlabel('marginal cost', fontsize=13)
                        t = 'marginal_costs'

                    plt.savefig(f'NPGA_{t}_{np}.pdf')

            elif asymmetric:


                plt.figure(figsize=(6,5))

                for key in model_dict:

                    pred = model_dict[key](cv).cpu().detach()
                    lo = key[7]
                    hi = key[8]
                    plt.plot(cv.cpu(), pred, marker='*', label=f'vs. vj ~ U[{lo},{hi}]')

                plt.legend()
                plt.title(f'{c_title}', fontsize=12)
                plt.ylabel('effort ei', fontsize=13)
                if use_valuations_result:
                    plt.xlabel('valuations', fontsize=13)
                    t = 'valuation vi ~ U([0.1, 1.1])'
                else:
                    plt.xlabel('marginal cost', fontsize=13)
                    t = 'marginal_costs'

                plt.savefig(f'NPGA_asym.pdf')

            else:

                plt.figure(figsize=(6,5))


                for key in model_dict:
                    pred = model_dict[key](cv).cpu().detach()
                    ifactor = key[1]
                    plt.plot(cv.cpu(), pred, marker='*', label=f'm={ifactor}')

                plt.legend()
                plt.title(f'Numerical solution of {c_title}', fontsize=14)
                plt.ylabel('effort/bid', fontsize=13)
                if use_valuations_result:
                    plt.xlabel('valuations', fontsize=13)
                    t = 'valuations'
                else:
                    plt.xlabel('marginal cost', fontsize=13)
                    t = 'marginal_costs'

                plt.savefig(f'NPGA_{c_title}_{t}.pdf')


    # Config # 
    gpus = [3, 3]

    # Generalized Tullock Contest
    use_valuations = [True, False]
    impact_factors = [0, 0.5, 1, 2, 5]
    c_type = "tullock_contest"
    c_title = "Tullock Contest"
    config = [use_valuations, impact_factors]
    #plot_config(config, c_type, c_title, gpus)


    # # Difference Form
    use_valuations = [True, False]
    impact_factors = [0.1, 1, 5, 10]
    c_type = "difference_contest"
    c_title = "Tullock Contest (Difference Form)"
    config = [use_valuations, impact_factors]
    #plot_config(config, c_type, c_title, gpus)


    # # Affiliated Setting
    use_valuations = [True]
    impact_factors = [1, 2, 3]
    corr_params = [0.3, 0.6, 0.9]
    c_type = "tullock_contest"
    c_title = "affilited Tullock Contest"
    config = [use_valuations, impact_factors, corr_params]
    #plot_config(config, c_type, c_title, gpus)

    # Adjusted cost function
    use_valuations = [True, False]
    impact_factors = [0.5, 1, 2]
    cost_types = ["exponent"]
    cost_params = [0.5, 1, 2]
    c_type = "tullock_contest"
    c_title = "Adjusted cost functions"
    config = [use_valuations, impact_factors, cost_types, cost_params]
    #plot_config(config, c_type, c_title, gpus)

    # Crowdsourcing Contests
    use_valuations = False
    impact_factors = [1.0, 0.8, 0.6]
    contestants = [3, 5]
    c_type = f'crowdsourcing'
    c_title = f'Crowdsourcing - 2 prizes, x contestants'
    config = [use_valuations, impact_factors, contestants, "crowdsourcing"]
    plot_config(config, c_type, c_title, gpus)

    # Asymmetric Contests
    use_valuations = [True]
    impact_factors = [1.0]
    priors_lo = [[0.1], [0.1, 1.1], [0.1, 2.1], [0.1, 3.1]]
    priors_hi = [[1.1], [1.1, 2.1], [1.1, 3.1], [1.1, 4.2]]
    c_type = f'tullock_lottery'
    c_title = f'Tullock lottery\nweak bidder vs. different stronger bidders'
    config = [use_valuations, impact_factors, priors_lo, priors_hi, "asymmetric"]
    #plot_config(config, c_type, c_title, gpus)