from BayesianOptimization.bayes_opt import BayesianOptimization

def black_box_function(x, y):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    #experiment_config, experiment_class = ConfigurationManager(experiment_type='single_item_uniform_symmetric', n_runs=2, n_epochs=3) \
    #    .get_config()
    #experiment = experiment_class(experiment_config)                                                             

    return -x ** 2 - (y - 1) ** 2 + 1


#print(black_box_function(2,1))
# Bounded region of parameter space
pbounds = {'x': (-100, 100), 'y': (-100, 100)}
pbounds = {'x': (0, 4), 'y': (-3, 3)}

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)

optimizer.maximize(init_points=5, n_iter=10)

#for i, res in enumerate(optimizer.res):
#    print("Iteration {}: \n\t{}".format(i, res))

print(optimizer.max)



#changing bounds
optimizer.set_bounds(new_bounds={"x": (-2, 3)})

#probe specific points before gp starts 
#Beware that the order has to be alphabetical. You can usee optimizer.space.keys for guidance
optimizer.probe(
    params={"x": 0.5, "y": 0.7},
    lazy=True, #meaning these points will be evaluated only the next time you call maximize
)

# %%
