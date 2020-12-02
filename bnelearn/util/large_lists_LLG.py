"""This module contains all possible solutions for the LLG setting to be checked."""
# row, column for ones
subsolutions  = [
    # 1. one bundle won
    [0, 0],
    [1, 1],
    [2, 2],
    [3, 3],
    [4, 4],
    [5, 5],
    [6, 6],
    [7, 7],
    [8, 8],
    # 2. two bundles won (-> must be single-item bundles 0, 1)
    # 2.1 won by different agents
    [9, 0],
    [9, 3],
    [10, 0],
    [10, 6],
    [11, 3],
    [11, 6],
    # 2.2 won by same agent
    [12, 0],
    [12, 1],
    [13, 3],
    [13, 4],
    [14, 6],
    [14, 7],
    ]

solutions_non_sparse = [
        # global bundle is won
        [0,0,1, 0,0,0, 0,0,0],
        [0,0,0, 0,0,1, 0,0,0],
        [0,0,0, 0,0,0, 0,0,1],
        # items or won individually
        [1,1,0, 0,0,0, 0,0,0],
        [1,0,0, 0,1,0, 0,0,0],
        [1,0,0, 0,0,0, 0,1,0],
        [0,1,0, 1,0,0, 0,0,0],
        [0,0,0, 1,1,0, 0,0,0],
        [0,0,0, 1,0,0, 0,1,0],
        [0,1,0, 0,0,0, 1,0,0],
        [0,0,0, 0,1,0, 1,0,0],
        [0,0,0, 0,0,0, 1,1,0],
    ]
