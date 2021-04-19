import os
import sys

import torch

# pylint: disable=wrong-import-position
sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'bnelearn'))

from bnelearn.experiment.configuration_manager import ConfigurationManager # pylint: disable=import-error

if __name__ == '__main__':
    log_root_dir = os.path.join(os.path.expanduser('~'), 'bnelearn', 'experiments')

    experiment_config, experiment_class = ConfigurationManager(experiment_type='single_item_uniform_symmetric', n_runs=10,
                                                                n_epochs=1000) \
            .set_setting(risk=1.1)\
            .set_logging(log_root_dir=log_root_dir, save_tb_events_to_csv_detailed=True)\
            .set_learning(pretrain_iters=5) \
            .set_logging(eval_batch_size=2**22).set_hardware(specific_gpu=5).get_config()

    log = {'dir': [], 'number_runs': [], 'epoch': [], 'lr':[], 'eps_rel': [], 'eps_rel_var': []}
    log['dir'].append(experiment_config.logging.experiment_dir)

    try:
        experiment = experiment_class(experiment_config)

        # Could only be done here and not inside Experiment itself while the checking depends on Experiment subclasses
        if ConfigurationManager.experiment_config_could_be_saved_properly(experiment_config):
            result = experiment.run()
        else:
            raise Exception('Unable to perform the correct serialization')

    except KeyboardInterrupt:
        print('\nKeyboardInterrupt: released memory after interruption')
        torch.cuda.empty_cache()
    
    print(result)
