"""
    Utilities for logging
"""

import pathlib
import socket
import time

from torch.utils.tensorboard import SummaryWriter

def get_experiment_writer(game: str, method: str, run_id: str or int = None, root_dir=None, **kwargs) -> SummaryWriter:
    """
        Creates and returns a torch.utils.tensorboard `SummaryWriter` to be used with a given experiment, performing
        validity checks to make sure logdir parameters are set as desired.

        IMPORTANT: you must close the writer when your experiment is done!

        Each experiment should create its own unique writer (and close it when done!), resulting logs will be written
        to the directory `root_dir/game/method/run_id` (see below for details)

        TODOs:
        --------
        * TODO: Implement automatic detection of namestrings of game objects (and possibly solution methods.)
        * TODO: Implement automatic capture of initial game/method hyperparams (if any)

        Parameters
        ----------
        game: str
            name of the game that the experiment is about, e.g. 'PrisonersDillemma'. (must be valid directory name)
        method: str
            name of the solution method being applied, e.g. 'FictitiousPlay'. (must bew a valid directory name)
        run_id (optional): str or int
            Name or identifier of current specific experiment run. E.g. 'low_temperature', '1', '5th experiment', etc.
            If ommitted, will use current system time.
        root_dir (optional): str or filepath
            Directory where all experiment results will be written to. On srvbichler14, this will default to
                /srv/bnelearn_experiments/
            On all other systems, must be explicitly supplied. (e.g. User home directory on windows etc)

        Returns
        -------
        writer: torch.utils.tensorboard.SummaryWriter
            A summaryWriter object that can be used to log
                - scalars
                - histograms
                - matplotlib figures
                - custom plots
                - images
                - audio
                - video
                - NN graphs
                - text
            either globally, or in each iteration.
            See documentation here: https://pytorch.org/docs/stable/tensorboard.html


        Examples
        -------------
        myGameParameter = ...
        myMethodParameter1, myMethodParameter2 = ...

        > game = MyGame(someGameParameter)
        > writer = get_experiment_writer('gameName', 'methodName')
        > writer.add_scalar('hyperparams/game/theGameParameter', myGameParameter, global_step=0)
        > writer.add_scalars(main_tag = 'hyperparams/method',
        >                    {'methodParam1': myMethodParameter1, 'methodParam2': myMethodParameter2},
        >                    global_step=0)
        >
        > for i in range(number_iterations):
        >     result1, result2 = methodIteration(myMethodParameter1, myMethodparameter2)
        >     writer.add_scalar('eval/result1', result1, i)
        >     writer.add_scalar('eval/result2', result2, i)
        >
        > writer.add_scalar('final_results/result1', result1, 0)
        >
        > writer.close()

    """

    # when root_dir is not specified, use default path on srvbichler14-server or
    # fail on different machines.
    # A root_dir given explicitly should exist, to prevent typos.
    if not root_dir:
        assert socket.gethostname() == 'srvbichler14', \
            'Default root_dir is only available on the GPU server! Please provide one manually.'

        root_dir = '/srv/bnelearn_experiments/'
    else: assert pathlib.Path(root_dir).exists(), 'Root directory does not exist!'

    # TODO: get game name from game object directly
    assert isinstance(game, str), "game should be a string for now. parsing directly from game object not implemented."
    assert isinstance(method, str), "Must provide name of method as a string."

    # if no run_id given, use current time
    if not run_id:
        run_id = time.strftime('%Y-%m-%d %H.%M.%S %A')

    # set log dir
    log_dir = pathlib.Path(root_dir) / str(game) / str(method) / run_id

    assert (not log_dir.exists()), "This run already exists!"

    writer = SummaryWriter(log_dir = log_dir, comment= "{}, {}".format(game, method), **kwargs)
    writer.add_text('experiment/game', game)
    writer.add_text('experiment/method', method)

    return writer
