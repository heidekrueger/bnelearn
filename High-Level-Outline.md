# Classes

### Bidder / Player
A player in a game. Holds some private information and a utility function. Has some strategy.

### Strategy
A strategy mapping from (private) information to an action (or distribution over actions). Usually a neural network. We’ll call an instance of a strategy a model.
Implemented as a pytorch.module


### Mechanism / Game
A game. For a given set of actions, defines an outcome.

### Environment
Class that manages the playing setup. It usually references and manages:
* A game/mechanism
* A set of Players/Bidders
    * 	Each player can evolve over time (i.e. update their strategy)
    *	For “dynamic” environments, players can be replaced / added over time. For ‘fixed’ environments, player identities are fixed
    * 	It’s possible for players to share a common strategy (i.e. one player’s update will affect the others)
* Logic to coordinate the above.

### Optimizer
An optimizer (using pytorch’s interface) that implements a strategy-update for a strategy. Can pertain to a single player or a set of players that use that strategy.


# Notebooks / Experiments / Main-Scripts

* Define and run an experiment with appropriate initializations of the classes above.
* Training loop
    * Tell optimizer to update current strategy. Optimizer calls the following steps:
        * Optimizer calls on environment to play the game (a `batch_size` number of times in parallel) to get the information it needs (i.e. rewards)
            * Environment makes players redraw their private valuations and gathers resulting actions from players
            * Environment runs the mechanism, determines outcomes.
        * Optimizer calls on its corresponding player to determine her utility for current iteration.
        * Optimizer updates corresponding strategy using information available.
    * possibly update scheduled hyperparameters
    * log statistics in tensorboard format, plot results
    * repeat