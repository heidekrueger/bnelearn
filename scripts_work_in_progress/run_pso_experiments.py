"""
Runs predefined experiments with individual parameters
"""
import os
import sys
import torch

# put bnelearn imports after this.
# pylint: disable=wrong-import-position


sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'bnelearn'))

from bnelearn.experiment.configuration_manager import ConfigurationManager  # pylint: disable=import-error

if __name__ == '__main__':
    # Path
    log_root_dir = os.path.join(os.path.expanduser('~'), 'bnelearn', 'experiments')
    ### Params ###
    # default setting: n_players=2, payment_rule='first_price',risk=1.0
    n_runs = 10
    n_epochs = 1000
    specific_gpu = 4
    eval_batch_size = 2 ** 22
    # PSO
    model_sharing = True
    pretrain = 500
    learner_type = 'PSOLearner'
    learner_hyperparams={
        'swarm_size': 30,
        'topology': 'von_neumann',
        'pretrain_deviation': 0.0,
        'inertia_weight': 0.729,
        'cognition_ratio': 1.49445,
        'social_ratio': 1.49445,
        'reevaluation_frequency': 10,
        'bound_handling': False,
        'velocity_clamping': True},
    optimizer_type='PSO'
    optimizer_hyperparams={}

    # NPGA learner
    #model_sharing = True
    #pretrain = 500
    #learner_type = 'ESPGLearner',
    #learner_hyperparams = {'population_size': 64,
    #                       'sigma': 1.,
    #                       'scale_sigma_by_model_size': True},
    #optimizer_type = 'adam',
    #optimizer_hyperparams = {'lr': 1e-3},

    experiment_names = []
    params_to_eval = []

    for param, name in zip(params_to_eval, experiment_names):
        log_dir = os.path.join(log_root_dir, name)
        experiment_config, experiment_class = ConfigurationManager(
            experiment_type='single_item_uniform_symmetric',
            n_runs=n_runs,
            n_epochs=n_epochs) \
        .set_hardware(specific_gpu=specific_gpu) \
        .set_logging(
            log_root_dir=log_root_dir,
            save_tb_events_to_csv_detailed=True,
            util_loss_batch_size=2 ** 10,
            util_loss_grid_size=2 ** 10,
            eval_batch_size=eval_batch_size) \
        .set_learning(
            model_sharing=model_sharing,
            pretrain_iters=pretrain,
            learner_type=learner_type,
            learner_hyperparams=learner_hyperparams,
            optimizer_type=optimizer_type,
            optimizer_hyperparams=optimizer_hyperparams) \
        .get_config()

        try:
            experiment = experiment_class(experiment_config)
            if ConfigurationManager.experiment_config_could_be_saved_properly(experiment_config):
                experiment.run()
            else:
                raise Exception('Unable to perform the correct serialization')

        except KeyboardInterrupt:
            print('\nKeyboardInterrupt: released memory after interruption')

        finally:
            torch.cuda.empty_cache()
