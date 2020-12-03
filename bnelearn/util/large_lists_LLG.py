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
    # 2.1 agents 0, 1 win
    [9, 0],
    [9, 4],
    [10, 1],
    [10, 3],
    # 2.1.2 agents 0, 2 win
    [11, 0],
    [11, 7],
    [12, 1],
    [12, 6],
    # 2.1.3 agents 1, 2 win
    [13, 3],
    [13, 7],
    [14, 4],
    [14, 6]
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
