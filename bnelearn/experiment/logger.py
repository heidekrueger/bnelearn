import os
import sys
import matplotlib.pyplot as plt


class Logger:
    def __init__(self, save_figure_to_disc_png: bool = True, save_figure_to_disc_svg: bool = True,
                 plot_epoch: int = 100, show_plot_inline: bool = True):
        root_path = os.path.join(os.path.expanduser('~'), 'bnelearn')
        if root_path not in sys.path:
            sys.path.append(root_path)

        self.logging_options = dict(
            log_root=os.path.join(root_path, 'experiments'),
            save_figure_to_disc_png=save_figure_to_disc_png,
            save_figure_to_disc_svg=save_figure_to_disc_svg,  # for publishing. better quality but a pain to work with
            plot_epoch=plot_epoch,
            show_plot_inline=show_plot_inline
        )


    ## Setup logging
    def log_once(self, writer, e):
        """Everything that should be logged only once on initialization."""
        # writer.add_scalar('debug/total_model_parameters', n_parameters, e)
        # writer.add_text('hyperparams/neural_net_spec', str(self.model), 0)
        # writer.add_scalar('debug/eval_batch_size', eval_batch_size, e)
        writer.add_graph(self.model, self.env.agents[0].valuations)

    def log_metrics(self, writer, e):
        writer.add_scalar('eval/utility', self.utility, e)
        writer.add_scalar('debug/norm_parameter_update', self.update_norm, e)
        writer.add_scalar('eval/utility_vs_bne', self.utility_vs_bne, e)
        writer.add_scalar('eval/epsilon_relative', self.epsilon_relative, e)
        writer.add_scalar('eval/epsilon_absolute', self.epsilon_absolute, e)
        writer.add_scalar('eval/L_2', self.L_2, e)
        writer.add_scalar('eval/L_inf', self.L_inf, e)

    # TODO: deferred until writing logger
    def log_hyperparams(self, writer, e):
        """Everything that should be logged on every learning_rate updates"""

    #     writer.add_scalar('hyperparams/batch_size', batch_size, e)
    #     writer.add_scalar('hyperparams/learning_rate', learning_rate, e)
    #     writer.add_scalar('hyperparams/momentum', momentum, e)
    #     writer.add_scalar('hyperparams/sigma', sigma, e)
    #     writer.add_scalar('hyperparams/n_perturbations', n_perturbations, e)
