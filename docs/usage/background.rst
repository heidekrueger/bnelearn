
==========================================
Auction Theory and Equilibrium Computation
==========================================

1. Background
============= 

The computation and analysis of equilibrium states in strategic settings is of utmost importance in the economic sciences. However, equilibria in many markets remain poorly understood. The computational complexity of finding Nash equilibria is known to be PPAD complete even for finite normal form games (NFG), a class that is considered to be hard unless P = NP `(Daskalakis et al. 2009) <https://epubs.siam.org/doi/abs/10.1137/070699652>`_. Despite these hardness results for the worst case, equilibrium computation has become tenable for NFGs of moderate sizes in recent years. Auction markets are of a particular interest to economists and policymakers, as a precise understanding of strategic behavior in proposed auction markets would be invaluable in the design and implementation market mechanisms with certain desiderata. However, the game theoretic properties of auctions are even less amiable to equilibrium computation: Auctions are commonly modeled as Bayesian games with continuous type and action spaces `(Harsanyi 1968) <https://pubsonline.informs.org/doi/abs/10.1287/mnsc.14.5.320>`_, resulting in an infinite-dimensional, functional space of possible strategies. Such games are no longer finite, so even Nash's famous theorem about the existence of equilibria `(Nash 1950) <https://www.pnas.org/doi/abs/10.1073/pnas.36.1.48>`_ no longer holds, and the complexity of finding Bayesian Nash equilibria in auctions may be NP-hard or worse depending on the specifics of the game. In fact, `Cai and Papadimitriou (2014) <https://dl.acm.org/doi/abs/10.1145/2600057.2602877?casa_token=07C6_r5kr9EAAAAA:UbyTRDZPdiPksTS3KjRpRTAzSb0lMi5AwsyR_4QPVG5Lnr_UTqq98fTuYfcyEEVwnOh7MK-JGg>`_ show that in certain combinatorial auctions, the computation of Bayesian Nash (BNE) is at least PP-hard (a complexity class above the polynomial hierarchy), and even certifying or approximating a BNE at a constant approximation bound remains NP-hard. The current understanding of strategic behavior in the field of auction theory therefore relies mainly on equilibria that have been analytically derived. Unfortunately, such results are scarce and elusive for all but a few simple settings. 

Most recently, however, there have been a range of first empirical successes in approximating Bayesian Nash equilibria in sealed-bid auctions using computational black-box methods that do not rely on manual mathematical analysis of specific settings `(Heymann and Mertikopoulos 2021, <https://arxiv.org/abs/2108.04506>`_ `Bichler et al. 2021, <https://www.nature.com/articles/s42256-021-00365-4>`_ `Bosshard et al. 2020, <https://www.jair.org/index.php/jair/article/view/11525>`_ `Li and Wellman 2020) <https://rezunli96.github.io/src/AAAI2021.pdf>`_ This suggests that the established hardness results may not apply to a wide range of auctions that are of practical interest and that it may be possible to characterize subclasses of continuous Bayesian games for which equilibrium computation may be feasible.

However, the literature on auctions is vast, with many variants of markets that are of interest. This has been a limiting factor for research in equilibrium computation, as each individual study has only been able to investigate new methods in a small number of relevant auction settings due to implementation complexity. Existing equilibrium learning methods rely on simulating very large numbers of auctions, which has required computational optimizations that have often been hand tailored to specific classes of auctions or even the specific setting at hand.

To facilitate future research in equilibrium computation in auctions, we propose consolidating a wide set of auction settings into a single benchmark suite with a common, modular, extensible and performant programming interface. To this end, we present our open-source package `bnelearn`, which provides a GPU-accelerated framework for equilibrium computation in sealed-bid auctions and related (Bayesian and complete-information) games. Using `bnelearn`, researchers working on novel equilibrium learning rules will have access to a wide selection of implemented auction settings with or without existing equilibria as well as to an ecosystem of metrics and tools that facilitate analysis. To the authors' knowledge, `bnelearn` comprises the largest and most complete suite of implementations of sealed-bid auctions and their known equilibria, collecting 

In addition to researchers in equilibrium computation, we expect that such a framework will also be useful to practitioners of auction theory in the economic sciences: Given the fact that existing approaches have been empirically demonstrated to converge to approximate equilibria in a wide range of auction settings, and (approximate) error bounds can be calculated for candidate solutions, `bnelearn` enables analyses of strategic behavior in markets that elude analytical equilibrium analysis. As an example, in `(Bichler et al. 2021) <https://www.nature.com/articles/s42256-021-00365-4>`_, we were empirically able to quantify the effects of correlation between bidders on revenue and economic efficiency in equilibrium of small combinatorial auctions with core-selecting payment rules.


2. Problem Statement
====================

2.1 Model of an Auction Game
----------------------------

We consider *sealed-bid* auctions as special cases of Bayesian games. In such games, an auctioneer aims to sell :math:`m` goods to :math:`n` competing buyers. These buyers each submit bids based on their private information. Based on these bids, the auctioneer then uses an *auction mechanism* to allocate the goods to the buyers and determine what prices to charge the winners. The most general case of such an auction game can be formalized as the tuple :math:`G = (n, m, \mathcal{V}, \mathcal{O}, F, \mathcal{A}, x, p, u)`, where

* :math:`n` is the number of participating *bidders* or *agents*. We denote the set of bidders by :math:`\mathcal{I} = \lbrace 1, \dots, n\rbrace` and use the letter :math:`i` to index it.
* :math:`m` is the number of *goods* to be sold. When goods are heterogenous, we further denote by :math:`\mathcal{K} = 2^{[m]}` the set of *bundles* of goods, and index it by :math:`k`. When goods are homogenous, we instead use :math:`\mathcal{K} = {[m]} \cup \{0\}` to describe the possible cardinalities of subsets of the goods.
* :math:`\mathcal{V} = \mathcal{V}_1 \times \dots \times \mathcal{V}_n` describes the set of possible *valuations* of the bidders: Each bidder :math:`i` may be potentially interested in a subset :math:`\mathcal{K}_i \subseteq \mathcal{K}` of the possible bundles. Writing :math:`K_i = \left\lvert \mathcal{K}_i \right\rvert` for it's cardinality, :math:`\mathcal{V}_i \subseteq \mathbb{R}^{K_i}_{+}` then is the set of possible *valuation vectors* for agent :math:`i`. For example, in an auction of two heterogenous items :math:`\{a, b\}`, we might have :math:`\mathcal{V}_i = \mathbb{R}^4_+` and a vector :math:`v_i = (0, 1, 2, 5)` would indicate agent :math:`i`'s valuations for winning the empty bundle, only item :math:`a`, only item :math:`b`, or both items, respectively. Note that in some cases, bidders may not directly observe their true valuations :math:`v_i`, but may only have access to partial information about them:
* :math:`\mathcal{O} = \mathcal{O}_1 \times \dots \times \mathcal{O}_n` describes the set of possible *signals* or observations of private information that the bidders have access to. In the *private values* model, where bidders have full information about their valuations, we have :math:`o_i = v_i`.
* :math:`F` is the cumulative density function of the joint *prior* distribution over bidders' types, given by tuples :math:`(o_i, v_i)` of observations and valuations: :math:`F: \mathcal V \times \mathcal O \rightarrow [0, 1]`. It's probability density function will be denoted by :math:`f` and we state no further assumptions on the prior, thus, allowing for arbitrary correlations. Its marginals are denoted by :math:`f_v`, :math:`f_{o_i}`, etc. and its conditionals by :math:`f_{v_i\vert o_i}`.
* :math:`\mathcal{A} = \mathcal{A}_1 \times \dots \times \mathcal{A}_n = \mathbb{R}^n_{\geq 0}` are the available actions or *bids* to the bidders. These must be decided on based on the strategy :math:`\beta_i` and the information they have available, namely their observations :math:`o_i`: :math:`\beta_i(o_i) = b_i`.
* TODO: Strategies?
* :math:`x = (x_1, \dots, x_n) \in \{0, 1\}^{|\mathcal K|}` and :math:`p = (p_1, \dots, p_n) \in \mathbb{R}^n` describe the allocations and the payments that are determined by the mechanism after bids :math:`b \in \mathcal{A}` have been reported. An allocation constitutes a partition of the :math:`m` items, where bidder :math:`i` is allocated the bundle :math:`x_i`. In the simplest case, the allocations would be chosen such that the seller revenue (the sum of all bidders' payments) is maximized when bidders pay what they report. This is known as the first-price sealed bid auction.
* :math:`u = (u_1, \dots, u_n)` then is the utility vector of the bidders, where bidder :math:`i`'s utility :math:`u_i(v_i, b)` depends on their own valuation but all bidders' actions. Assuming the other bidders follow :math:`\beta`, bidder :math:`i`'s *interim utility* is then defined as the expected utility of choosing a bid :math:`b_i` conditioned on their observation :math:`o_i`:

    .. math::
        \overline{u}_i(o,b_i,\beta_{-i}) = \mathbb{E}_{v_i,o_{-i}|o_i}\left[u_i(v_i, b_i,\beta_{-i}(o_{-i}))\right].

    Accordingly, the interim *utility loss* :math:`\overline \ell` that is incurred by not playing a best response is:

    .. math::
        \overline \ell (o; b_i, \beta_{-i}) = \sup_{b'_i \in \mathcal A_i} \overline u_i(o_i, b'_i, \beta_{-i}) -\overline u_i(o_i, b_i, \beta_{-i}).

Furthermore, the *ex-ante utility* is defined as :math:`\tilde{u}_i(\beta_i,\beta_{-i})=\mathbb{E}_{o_i \sim f_{o_i}} [\overline{u}_i(o_i, \beta_{i}(o_i), \beta_{-i})]`, and the *ex-ante loss* :math:`\tilde \ell_i(\beta_i, \beta_{-i})`.

The question to be answered now is: "What is the optimal strategy profile for the bidders?" The most common the solution concept for this question is the so-called Bayes-Nash equilibrium: An *(interim) :math:`\epsilon`-Bayes-Nash equilibrium (:math:`\epsilon`-BNE)* is a strategy profile :math:`\beta^* = (\beta^*_1, \dots, \beta^*_n)` such that no agent can improve their own utility by more than :math:`\epsilon \geq 0` by unilaterally deviating from :math:`\beta^*`:

.. math::
    \forall\ i\in\mathcal I, o_i \in \mathcal O_i: \quad \overline{\ell}_i\left(o_i; \beta^*_i(o_i), \beta^*_{-i}\right)  \leq  \epsilon.

For :math:`\epsilon = 0`, the BNE is called _exact_, or the :math:`\epsilon`-prefix is simply dropped. The *ex-ante :math:`\epsilon`-BNE* is defined analogously.


2.2 Related Literature
----------------------

There is a large body of work on learning in games. However, the closer one comes to the modeling of auction games---continues type- and action spaces, general-sum games, non differentiable utility functions---less and less theoretical results or just heuristic learning approaches exist.

For rather small auction settings and under further assumptions, one is able to derive equilibria analytically for specific settings. Some fundamental results are listed by `(Krishna 2009) <https://www.elsevier.com/books/auction-theory/krishna/978-0-12-374507-1>`_.


Methods for Equilibrium Learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Equilibrium learning is now concerned with finding an optimal strategy profile for the agents. Where optimality is generally defined as the profile being a Bayes-Nash equilibrium. It has been shown that finding exact equilibria for these types of games with continuous state and action spaces is a computationally hard problem. However, we were able to provably approximate equilibria in auction games under symmetry assumptions `(Bichler et al. 2021) <https://www.nature.com/articles/s42256-021-00365-4>`_.

Earlier approaches to compute auction game equilibria approximately either comprised solving the set of differential equations resulting from the first order conditions of simultaneous maximization of the bidders' payoffs `(Marshall et al. 1994, <https://www.sciencedirect.com/science/article/pii/S0899825684710451>`_ `Bajari 2001) <https://link.springer.com/article/10.1007/PL00004128>`_, or of restricting the action space, usually by discretization `(Athey 2001) <https://onlinelibrary.wiley.com/doi/abs/10.1111/1468-0262.00223>`_. `Armantier et al. (2088) <https://onlinelibrary.wiley.com/doi/abs/10.1002/jae.1040>`_ introduced a general BNE computation method that is based on expressing the Bayesian game as the limit of a sequence of complete information games. Defining this sequence, however, also requires setting specific analysis. More recently, research in machine learning contributed to learning good bidding strategies in repeated revenue maximizing auctions `(Nedelec et al. 2019) <http://proceedings.mlr.press/v97/nedelec19a.html>`_. `Bosshard et al. (2017, 2020) <https://www.jair.org/index.php/jair/article/view/11525>`_ were first to compute equilibria in more complex combinatorial auctions. Their approach explicitly computes point-wise best responses in a fine grained linearization of the strategy space via sophisticated Monte-Carlo integration. Assuming independent priors and risk neutral utility functions, their verification method guarantees an upper bound :math:`\epsilon` on the interim loss in utility, thus provably finding an :math:`\epsilon`-BNE.
