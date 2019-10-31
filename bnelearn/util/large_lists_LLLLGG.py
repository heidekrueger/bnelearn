subsolutions  = [
            [0, 99, 99, 99],
            [1, 99, 99, 99],
            [2, 99, 99, 99],
            [3, 99, 99, 99],
            [4, 99, 99, 99],
            [5, 99, 99, 99],
            [6, 99, 99, 99],
            [7, 99, 99, 99],
            [8, 99, 99, 99],
            [9, 99, 99, 99],
            [10, 99, 99, 99],
            [11, 99, 99, 99],
            [0,2, 99, 99],
            [0,3, 99, 99],
            [0,4, 99, 99],
            [0,5, 99, 99],
            [0,6, 99, 99],
            [0,9, 99, 99],
            [0,10, 99, 99],
            [1,3, 99, 99],
            [1,4, 99, 99],
            [1,5, 99, 99],
            [1,6, 99, 99],
            [1,7, 99, 99],
            [1,9, 99, 99],
            [2,4, 99, 99],
            [2,5, 99, 99],
            [2,6, 99, 99],
            [2,7, 99, 99],
            [2,9, 99, 99],
            [2,11, 99, 99],
            [3,5, 99, 99],
            [3,6, 99, 99],
            [3,7, 99, 99],
            [3,11, 99, 99],
            [4,6, 99, 99],
            [4,7, 99, 99],
            [4,8, 99, 99],
            [4,11, 99, 99],
            [5,7, 99, 99],
            [5,8, 99, 99],
            [6,8, 99, 99],
            [6,10, 99, 99],
            [7,10, 99, 99],
            [0,2,4, 99],
            [0,2,5, 99],
            [0,2,6, 99],
            [0,2,9, 99],
            [0,3,5, 99],
            [0,3,6, 99],
            [0,4,6, 99],
            [0,6,10, 99],
            [1,3,5, 99],
            [1,3,6, 99],
            [1,3,7, 99],
            [1,4,6, 99],
            [1,4,7, 99],
            [1,5,7, 99],
            [2,4,6, 99],
            [2,4,7, 99],
            [2,4,11, 99],
            [2,5,7, 99],
            [3,5,7, 99],
            [4,6,8, 99],
            [0,2,4,6],
            [1,3,5,7]
            ]

solutions_sparse = [
            # 4 local bidders win
            [0, 2, 4, 6], 	# AB CD EF GH
            [1, 3, 5, 7], 	# BC DE FG HA
            # a global bidder wins (2 possibilities for each global bundle)
            [4, 6, 8, 99],     # EF GH ABCD
            [0, 2, 9, 99],    	# AB CD EFGH
            [0, 6, 10, 99],   	# AB GH CDEF
            [2, 4, 11, 99],    # CD EF GHAB
            [5, 8, 99, 99],        # FG ABCD
            [1, 9, 99, 99],		# BC EFGH
            [7, 10, 99, 99],       # HA CDEF
            [3, 11, 99, 99],       # DE GHAB
            # 3 locals win. This implies that 2 locals have adjacent bundles, and one doesn't.
            # there are 8 possibilities (choose bundle that has no adjacent bundles, the rest is determined)
            [0, 2, 5, 99],    	# AB CD FG
            [2, 4, 7, 99],     # CD EF HA
            [1, 4, 6, 99],		# BC EF GH
            [0, 3, 6, 99],    	# AB DE GH
            [0, 3, 5, 99],    	# AB DE FG
            [2, 5, 7, 99],     # CD FG HA
            [1, 3, 6, 99],		# BC DE GH
            [1, 4, 7, 99],		# BC EF HA
            ]

solutions_non_sparse = [
            # 4 local bidders win
            [1,0,1,0, 1,0,1,0, 0,0,0,0], # AB CD EF GH
            [0,1,0,1, 0,1,0,1, 0,0,0,0], # BC DE FG HA
            # a global bidder wins (2 possibilities for each global bundle)
            [0,0,0,0, 1,0,1,0, 1,0,0,0], # EF GH ABCD
            [1,0,1,0, 0,0,0,0, 0,1,0,0], # AB CD EFGH
            [1,0,0,0, 0,0,1,0, 0,0,1,0], # AB GH CDEF
            [0,0,1,0, 1,0,0,0, 0,0,0,1], # CD EF GHAB
            [0,0,0,0, 0,1,0,0, 1,0,0,0], # FG ABCD
            [0,1,0,0, 0,0,0,0, 0,1,0,0], # BC EFGH
            [0,0,0,0, 0,0,0,1, 0,0,1,0], # HA CDEF
            [0,0,0,1, 0,0,0,0, 0,0,0,1], # DE GHAB
            # 3 locals win. This implies that 2 locals have adjacent bundles, and one doesn't.
            # there are 8 possibilities (choose bundle that has no adjacent bundles, the rest is determined)
            [1,0,1,0, 0,1,0,0, 0,0,0,0], # AB CD FG
            [0,0,1,0, 1,0,0,1, 0,0,0,0], # CD EF HA
            [0,1,0,0, 1,0,1,0, 0,0,0,0], # BC EF GH
            [1,0,0,1, 0,0,1,0, 0,0,0,0], # AB DE GH
            [1,0,0,1, 0,1,0,0, 0,0,0,0], # AB DE FG
            [0,0,1,0, 0,1,0,1, 0,0,0,0], # CD FG HA
            [0,1,0,1, 0,0,1,0, 0,0,0,0], # BC DE GH
            [0,1,0,0, 1,0,0,1, 0,0,0,0], # BC EF HA
        ]