"""This module contains utilities for logging of experiments"""
import os
import pickle
import subprocess

from typing import List
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator, STORE_EVERYTHING_SIZE_GUIDANCE

import pkg_resources

from bnelearn.bidder import Bidder
from bnelearn.experiment.configurations import *


def read_bne_utility_database(exp: 'Experiment'):
    """Check if this setting's BNE has been saved to disk before.

    Args:
        exp: Experiment

    Returns:
        db_batch_size: int
            sample size of a DB entry if found, else -1
        db_bne_utility: List[n_players]
            list of the saved BNE utilites
    """
    file_path = pkg_resources.resource_filename(__name__, 'bne_database.csv')
    bne_database = pd.read_csv(file_path)

    # see if we already have a sample
    setting_database = bne_database[
        (bne_database.experiment_class == str(type(exp))) &
        (bne_database.payment_rule == exp.payment_rule) &
        (bne_database.correlation == exp.correlation) &
        (bne_database.risk == exp.risk)
    ]

    # 1. no entry found for this exp
    if len(setting_database) == 0:
        return -1, None

    # 2. found entry
    else:
        return setting_database['batch_size'].tolist()[0], setting_database.bne_utilities.tolist()


def write_bne_utility_database(exp: 'Experiment', bne_utilities_sampled: list):
    """Write the sampled BNE utilities to disk.

    Args:
        exp: Experiment
        bne_utilities_sampled: list
            BNE utilites that are to be writen to disk
    """
    file_path = pkg_resources.resource_filename(__name__, 'bne_database.csv')
    bne_database = pd.read_csv(file_path)

    bne_env = exp.bne_env if not isinstance(exp.bne_env, list) \
        else exp.bne_env[0]

    # see if we already have a sample
    setting_database = bne_database[
        (bne_database.experiment_class == str(type(exp))) &
        (bne_database.payment_rule == exp.payment_rule) &
        (bne_database.correlation == exp.correlation) &
        (bne_database.risk == exp.risk)
    ]

    # No entry found: make new db entry
    if len(setting_database) == 0:
        for player_position in [agent.player_position for agent in exp.bne_env.agents]:
            bne_database = bne_database.append(
                {
                    'experiment_class': str(type(exp)),
                    'payment_rule':     exp.payment_rule,
                    'correlation':      exp.correlation,
                    'risk':             exp.risk,
                    'player_position':  player_position,
                    'batch_size':       bne_env.batch_size,
                    'bne_utilities':    bne_utilities_sampled[player_position].item()
                },
                ignore_index=True
            )

    # Overwrite database entry
    else:
        for player_position in [agent.player_position for agent in exp.bne_env.agents]:
            bne_database.loc[
                (bne_database.experiment_class == str(type(exp))) &
                (bne_database.payment_rule == exp.payment_rule) &
                (bne_database.correlation == exp.correlation) &
                (bne_database.risk == exp.risk) &
                (bne_database.player_position == player_position)
            ] = [[str(type(exp)), exp.payment_rule, exp.correlation, exp.risk,
                  player_position, bne_env.batch_size,
                  bne_utilities_sampled[player_position].item()]]

    bne_database.to_csv(file_path, index=False)
