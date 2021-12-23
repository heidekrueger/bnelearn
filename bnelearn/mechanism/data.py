"""This module contains static data about all possible solutions
   in the LLG and LLLLGG combinatorial auctions, which may constitute efficient allocations."""

from abc import ABC, abstractmethod
from typing import List
import torch


class LLGData():
    """Contains static data about legal allocations in the LLG setting."""
    bundles = [[0], [1], [0,1]]

    # a 1 in position (3*i + k) indicates that player i gets bundle k.
    legal_allocations_dense = [
        # global bundle is won
        [0,0,1, 0,0,0, 0,0,0],
        [0,0,0, 0,0,1, 0,0,0],
        [0,0,0, 0,0,0, 0,0,1],
        # items are won individually
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

    # row, column for ones in the above representation
    legal_allocations_sparse  = [
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
        [ 9, 0], [ 9, 4],
        [10, 1], [10, 3],
        [11, 0], [11, 1],
        # 2.1.2 agents 0, 2 win
        [12, 0], [12, 7],
        [13, 1], [13, 6],
        [14, 3], [14, 4],
        # 2.1.3 agents 1, 2 win
        [15, 3], [15, 7],
        [16, 4], [16, 6],
        [17, 6], [17, 7]
        ]


class CombinatorialAuctionData(ABC):
    """Wraper class to represent static data of a large
    combinatorial auction.
    """
    @classmethod
    @property
    @abstractmethod
    def n_bundles(cls) -> int:
        pass

    @classmethod
    @property
    @abstractmethod
    def n_legal_allocations(cls) -> int:
        pass

    @classmethod
    @property
    @abstractmethod
    def _player_bundles(cls) -> List[List[int]]:
        pass

    @classmethod
    @property
    @abstractmethod
    def _efficient_allocations_dense(cls) -> List[List[int]]:
        pass

    @classmethod
    @property
    @abstractmethod
    def _efficient_allocations_semisparse(cls) -> List[List[int]]:
        pass

    @classmethod
    @property
    @abstractmethod
    def _legal_allocations_sparse(cls) -> List[List[int]]:
        pass

    @classmethod
    def player_bundles(cls, device='cpu') -> torch.Tensor:
        "returns the player bundle matching as a torch.Tensor"
        return torch.tensor(cls._player_bundles, device=device, dtype=torch.long)


    @classmethod
    def efficient_allocations_dense(cls, device='cpu') -> torch.Tensor:
        """Returns the possible efficient allocations as a dense tensor"""
        return torch.tensor(cls._efficient_allocations_dense,
                            dtype=torch.float,device=device)

    @classmethod
    def legal_allocations_sparse(cls, device='cpu') -> torch.Tensor:
        """returns the sparse index representation of legal allocations as
        a torch.Tensor."""
        return torch.tensor(cls._legal_allocations_sparse, device=device)

    @classmethod
    def legal_allocations_dense(cls, device='cpu') -> torch.Tensor:
        """returns a dense torch tensor of all possible legal allocations
        on the desired device.
        Output shape: n_legal_allocations x n_bundles
        """
        sparse = cls._legal_allocations_sparse
        n_allocations = sparse[-1][0] + 1 # highest row index + 1
        dense = torch.sparse_coo_tensor(
            torch.tensor(sparse, device=device).t(),
            torch.ones(len(sparse), device=device),
            [n_allocations, cls.n_bundles]
        ).to_dense()
        return dense

class LLLLGGData(CombinatorialAuctionData):
    """Static data about legal and possibly efficient allocations in the LLLLGG
    setting. For details about the representation, see
    Bosshard et al. (2019), https://arxiv.org/abs/1812.01955.
    """

    ###### possibly efficient allocations ######
    n_bundles = 12
    n_legal_allocations = 66

    _player_bundles = [
        # which bundle does each player demand?
        [0, 1],  #L1: AB, BC
        [2, 3],  #L2: CD, DE
        [4, 5],  #L3: EF, FG
        [6, 7],  #L4: GH, HA
        [8, 9],  #G1: ABCD, EFGH
        [10, 11] #G2: CDEF, GHAB
    ]

    _efficient_allocations_dense = [
        # rows: possible solutions that may be efficient
        # columns: allocated bundle (deterministic to one of the bidders due to the
        # demand structure)
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

    efficient_allocations_semisparse: List[List[int]] = [
        # each row represents a possibly efficient allocation.
        # indices are the indices of allocated bundles in the dense matrix above.

        # 4 local bidders win
        [0, 2, 4, 6], 	# AB CD EF GH
        [1, 3, 5, 7], 	# BC DE FG HA
        # a global bidder wins (2 possibilities for each global bundle)
        [4, 6, 8],     # EF GH ABCD
        [0, 2, 9],    	# AB CD EFGH
        [0, 6, 10],   	# AB GH CDEF
        [2, 4, 11],    # CD EF GHAB
        [5, 8],        # FG ABCD
        [1, 9],		# BC EFGH
        [7, 10],       # HA CDEF
        [3, 11],       # DE GHAB
        # 3 locals win. This implies that 2 locals have adjacent bundles, and one doesn't.
        # there are 8 possibilities (choose bundle that has no adjacent bundles, the rest is determined)
        [0, 2, 5],    	# AB CD FG
        [2, 4, 7],     # CD EF HA
        [1, 4, 6],		# BC EF GH
        [0, 3, 6],    	# AB DE GH
        [0, 3, 5],    	# AB DE FG
        [2, 5, 7],     # CD FG HA
        [1, 3, 6],		# BC DE GH
        [1, 4, 7],		# BC EF HA
        ]

    ###### ALL legal allocations #####

    # all legal allocations in sparse format (includes inefficient outcomes,
    # e.g. where not all items are allocated. These are necessary to solve
    # restricted subgames.)
    # Determines a sparse 66 x 12 matrix.
    # An entry (i,j) determines that bundle j is allocated in solution i.

    _legal_allocations_sparse: List[List[int]]  = [
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 3],
        [4, 4],
        [5, 5],
        [6, 6],
        [7, 7],
        [8, 8],
        [9, 9],
        [10, 10],
        [11, 11],
        [12, 0],  [12, 2],
        [13, 0],  [13, 3],
        [14, 0],  [14, 4],
        [15, 0],  [15, 5],
        [16, 0],  [16, 6],
        [17, 0],  [17, 9],
        [18, 0],  [18, 10],
        [19, 1],  [19, 3],
        [20, 1],  [20, 4],
        [21, 1],  [21, 5],
        [22, 1],  [22, 6],
        [23, 1],  [23, 7],
        [24, 1],  [24, 9],
        [25, 2],  [25, 4],
        [26, 2],  [26, 5],
        [27, 2],  [27, 6],
        [28, 2],  [28, 7],
        [29, 2],  [29, 9],
        [30, 2],  [30, 11],
        [31, 3],  [31, 5],
        [32, 3],  [32, 6],
        [33, 3],  [33, 7],
        [34, 3],  [34, 11],
        [35, 4],  [35, 6],
        [36, 4],  [36, 7],
        [37, 4],  [37, 8],
        [38, 4],  [38, 11],
        [39, 5],  [39, 7],
        [40, 5],  [40, 8],
        [41, 6],  [41, 8],
        [42, 6],  [42, 10],
        [43, 7],  [43, 10],
        [44, 0],  [44, 2],  [44, 4],
        [45, 0],  [45, 2],  [45, 5],
        [46, 0],  [46, 2],  [46, 6],
        [47, 0],  [47, 2],  [47, 9],
        [48, 0],  [48, 3],  [48, 5],
        [49, 0],  [49, 3],  [49, 6],
        [50, 0],  [50, 4],  [50, 6],
        [51, 0],  [51, 6],  [51, 10],
        [52, 1],  [52, 3],  [52, 5],
        [53, 1],  [53, 3],  [53, 6],
        [54, 1],  [54, 3],  [54, 7],
        [55, 1],  [55, 4],  [55, 6],
        [56, 1],  [56, 4],  [56, 7],
        [57, 1],  [57, 5],  [57, 7],
        [58, 2],  [58, 4],  [58, 6],
        [59, 2],  [59, 4],  [59, 7],
        [60, 2],  [60, 4],  [60, 11],
        [61, 2],  [61, 5],  [61, 7],
        [62, 3],  [62, 5],  [62, 7],
        [63, 4],  [63, 6],  [63, 8],
        [64, 0],  [64, 2],  [64, 4],  [64, 6],
        [65, 1],  [65, 3],  [65, 5],  [65 ,7]
    ]

class LLLLRRGData(CombinatorialAuctionData):
    """Static data about legal and possibly efficient allocations in the LLLLRRG
    setting. This extends the LLLLGG setting by adding a 7th, "superglobal"
    player, who is interested in the bundle of all 8 items.

    In this setting, we'll call the new player 'global', and the players
    interested in 4-item-bundles (R in LLLLRRG, G in LLLLGG) 'regional'.
    """

    ###### possibly efficient allocations ######
    n_bundles = 14 # 13 real bundles and an empty-pseudobundle
    n_legal_allocations = 67 # those in LLLLGG and [12]

    _player_bundles = [
        # which bundle does each player demand?
        [0, 1],  #L1: AB, BC
        [2, 3],  #L2: CD, DE
        [4, 5],  #L3: EF, FG
        [6, 7],  #L4: GH, HA
        [8, 9],  #R1: ABCD, EFGH
        [10, 11], #R2: CDEF, GHAB,
        [12, 13]  #G: 12: ABCDEFGH, 13 is an empty pseudo-bundle
    ]

    #dense allocations:
    # - either an allocation from LLLLGG (with added 0 in last bundle)
    # - or allocation to the Global player
    _efficient_allocations_dense = [
        a + [0, 0] for a in LLLLGGData._efficient_allocations_dense
    ] + [*[0]*12, 1, 0] # ABCDEFGH allocated to G

    efficient_allocations_semisparse = \
        LLLLGGData.efficient_allocations_semisparse + [12]

    _legal_allocations_sparse  = LLLLGGData._legal_allocations_sparse + [66, 12]

    
