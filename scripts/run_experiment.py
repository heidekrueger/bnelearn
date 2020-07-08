import os
import sys
import torch

sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'bnelearn'))

#pylint: disable=wrong-import-position
from bnelearn.experiment.configuration_manager import ConfigurationManager 

if __name__ == '__main__':
    '''
    Runs predefined experiments with individual parameters
    fire.Fire() asks you to decide for one of the experiments defined above
    by writing its name and define the required (and optional) parameters
    e.g.:
        experiment.py single_item_uniform_symmetric 1 20 [2,3] 'first_price'

    alternatively instead of fire.Fire() use, e.g.:
        single_item_uniform_symmetric(1,20,[2,3],'first_price')

    '''
    # running_configuration, logging_configuration, experiment_configuration, experiment_class = \
    #     fire.Fire()

    # Run from a file
    # experiment_config = logging.get_experiment_config_from_configurations_log()
    # experiment_class = ConfigurationManager.get_class_by_experiment_type(experiment_config.experiment_class)

    # Well, path is user-specific
    log_root_dir = os.path.join(os.path.expanduser('~'), 'bnelearn', 'experiments')

    # ToDo reset all the commented out settings to the same number of runs and epochs as before
    #experiment_config, experiment_class = ConfigurationManager(experiment_type='single_item_uniform_symmetric') \
    #    .get_config(save_tb_events_to_csv_detailed=True, log_root_dir=log_root_dir, n_runs=1, n_epochs=200)
    # experiment_config, experiment_class = ConfigurationManager(experiment_type='single_item_gaussian_symmetric') \
    #     .get_config(log_root_dir=log_root_dir)

    # All three next experiments get AssertionError: scalar should be 0D
    # experiment_config, experiment_class = \
    #    ConfigurationManager(experiment_type='single_item_asymmetric_uniform_overlapping') \
    #    .get_config(log_root_dir=log_root_dir)
    # experiment_config, experiment_class = \
    #     ConfigurationManager(experiment_type='single_item_asymmetric_uniform_disjunct') \
    #     .get_config(log_root_dir=log_root_dir)
    # experiment_config, experiment_class = ConfigurationManager(experiment_type='llg') \
    #     .with_correlation(gamma=0.0) \
    #     .get_config(log_root_dir=log_root_dir,
    #                 n_runs=1, n_epochs=100, specific_gpu=1,
    #                 # payment_rule='proxy',
    #                 )


    # experiment_config, experiment_class = ConfigurationManager(experiment_type='llllgg') \
    #     .get_config(log_root_dir=log_root_dir)

    experiment_config, experiment_class = ConfigurationManager(experiment_type='multiunit') \
        .get_config(log_root_dir=log_root_dir)
    # experiment_config, experiment_class = ConfigurationManager(experiment_type='splitaward')\
    #     .get_config(log_root_dir=log_root_dir)
    try:
        experiment = experiment_class(experiment_config)
        #TODO: this is a short term fix - we can only determine whether BNE exists once experiment has been initialized.
        # Medium Term -->  Set 'opt logging in experiment itself.
        if experiment.known_bne:
            experiment.logging.log_metrics = {
                'opt': True,
                'l2': True,
                'util_loss': True
            }

        # Could only be done here and not inside Experiment itself while the checking depends on Experiment subclasses
        if ConfigurationManager.experiment_config_could_be_serialized_properly(experiment_config):
            experiment.run()
        else:
            raise Exception('Unable to perform the correct serialization')
    except KeyboardInterrupt:
        print('\nKeyboardInterrupt: released memory after interruption')
        torch.cuda.empty_cache()
