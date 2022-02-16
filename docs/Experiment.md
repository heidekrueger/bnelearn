# Documentation: Experiments

## Design goal:
* Have a very simple interface to create _all_ possible experiment settings from json, command line interface, dict, etc.
* Should allow presets (possibly with individual values changed), as well as fully specified experiments.


## The `Experiment` class

Immutable members
* params describing the experiment
* plotting and logging functions
* BNE env if available

Mutable State
* learning environment (models, bidders, learners with their own internal state)
* logging state

### Object Creation

* run_script calls a `Builder`/`Factory` (which one?) that parses argument input and creates the appropriate `Experiment` Object by calling the constructor of the appropriate subclass.
* Constructors work strictly bottom-up. No implementation of anything optional in parent classes. `super()` constructors should not rely on any additional info from the subclasses. Anything that's overwritten/changed from base class instantiation should be changed in the subclass.
* The base class should retain some information about whether some optional features are implemented. (e.g. interfaces). Ideally, I'd like to implement this using interfaces/mixins but I'm not sure about the most pythonic way to do this. For now, we'll give the top level class a flag like `_has_known_bne` that can be called at runtime to determine whether to execute option-specific code (such as calculating appropriate metrics).



### Inheritcance
* Specific settings inherit from more abstract settings
* BNEs/optimal strategies should be implemented on the highest possible level. In some cases, a more specific setting will have a _simplified_ form of the BNE already available at a higher level (e.g.: `UniformSymmetricFPSB` implements the same BNE already known from `SymmetricFPSB`, but the more specific form does not rely on numeric integration and is computationally much cheaper). In such cases, the child should replace the parent function.

### Known BNE