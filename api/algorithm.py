# Pykumu - https://github.com/sailuh/pykumu
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Algorithm execution for structure learning.

This module provides functions for running causal search algorithms
(FGES, BOSS) using the Tetrad library via JPype.
"""

def algorithm_fges(data, params, score, knowledge, symmetric_first_step=False, max_degree=-1,
             parallelized=False, faithfulness_assumed=False, num_threads=5):
    """Implements the Fast Greedy Equivalence Search (FGES) algorithm.

    This is an implementation of the Greedy Equivalence Search algorithm,
    originally due to Chris Meek but developed significantly by Max Chickering.
    FGES uses with some optimizations that allow it to scale accurately to
    thousands of variables accurately for the sparse case. The reference for FGES is this:
    The reference for Chickering's GES is this:
    Chickering (2002) "Optimal structure identification with greedy search"
    Journal of Machine Learning Research.
    FGES works for the continuous case, the discrete case,
    and the mixed continuous/discrete case, so long as a BIC score is available
    for the type of data in question.
    To speed things up, it has been assumed that variables X and Y with
    zero correlation do not correspond to edges in the graph. This is a
    restricted form of the heuristic speedup assumption, something GES does not assume.
    This heuristic speedup assumption needs to be explicitly turned on using setHeuristicSpeedup(true).
    Also, edges to be added or remove from the graph in the forward or backward phase,
    respectively are cached, together with the ancillary information needed to do the
    additions or removals, to reduce rescoring.
    A number of other optimizations were also. See code for details.
    This class is configured to respect knowledge of forbidden and required edges,
    including knowledge of temporal tiers.
    For more details, see: https://www.phil.cmu.edu/tetrad-javadocs/7.6.0/edu/cmu/tetrad/search/Fges.html

    :param data: Tetrad data object returned by \code{\link{data.transform_pandasdf_to_tetrad_boxdataset}}.
    :param params: Tetrad Parameters object returned by \code{\link{data.transform_pandasdf_to_tetrad_boxdataset}} (and configured by \code{\link{bootstrapping.bootstrapping}} and \code{\link{score.use_sem_bic}}).
    :param score: Tetrad score object returned by \code{\link{score.use_sem_bic}}.
    :param knowledge: Tetrad Knowledge object returned by \code{\link{data.transform_pandasdf_to_tetrad_boxdataset}} or replaced via \code{\link{knowledge.parse_knowledge_txt}}.
    :param symmetric_first_step: TRUE if the first step step for FGES should do scoring for both X->Y and Y->X
    :param max_degree: Integer. The maximum degree of the graph (min = -1)
        from different random starting permutations. The model with the most
        optimal BIC score will be selected. Random after the first. Defaults to 1.
    :param parallelized: TRUE if the search should be parallelized
    :param faithfulness_assumed: TRUE if (one edge) faithfulness should be assumed
    :param num_threads: The number of threads (>= 1) to use for the search. Defaults to 5.
    :returns: dict with 'graph' (Java graph object) and 'bootstrap_graphs'

    :references: Ramsey, J., Glymour, M., Sanchez-Romero, R., & Glymour, C. (2017).
        A million variables and more: the fast greedy equivalence search algorithm for
        learning high-dimensional graphical causal models, with an application to functional
        magnetic resonance images. International journal of data science and analytics, 3, 121-129.
    """
    import edu.cmu.tetrad.algcomparison.algorithm.oracle.cpdag as cpdag
    from edu.cmu.tetrad.util import Params

    alg = cpdag.Fges(score)
    alg.setKnowledge(knowledge)

    params.set(Params.SYMMETRIC_FIRST_STEP, symmetric_first_step)
    params.set(Params.MAX_DEGREE, max_degree)
    params.set(Params.PARALLELIZED, parallelized)
    params.set(Params.FAITHFULNESS_ASSUMED, faithfulness_assumed)
    params.set(Params.NUM_THREADS, num_threads)

    graph = alg.search(data, params)
    bootstrap_graphs = alg.getBootstrapGraphs()

    return {"graph": graph, "bootstrap_graphs": bootstrap_graphs}


def algorithm_boss(data, params, score, knowledge, num_starts=1, use_bes=False, time_lag=0,
             use_data_order=True, output_cpdag=True, num_threads=5):
    """Implements the BOSS (Best Order Score Search) algorithm.

    BOSS (Best Order Score Search) is an algorithm that, like GRaSP,
    generalizes and extends the GSP (Greedy Sparsest Permutation) algorithm.
    It has been tested to 1000 variables with an average degree of 20 and gives
    near perfect precisions and recalls for N = 10,000
    (with recall that drop to 0.9 for N = 1000).

    The algorithms works by building DAGs given permutations in ways similar
    to those described in Raskutti and Uhler and Solus et al. (see references below)

    Knowledge of forbidden edges and required edges may be used with this algorithm.
    Also, knowledge of tiers may be used. If tiered knowledge is supplied,
    the algorithm will analyze the tiers in order, so that the time required
    for the algorithm is linear in the number of tiers.

    For more details, see: https://www.phil.cmu.edu/tetrad-javadocs/7.4.0/edu/cmu/tetrad/search/Boss.html
    and https://cmu-phil.github.io/tetrad/manual/#boss

    :param data: Tetrad data object returned by \code{\link{data.transform_pandasdf_to_tetrad_boxdataset}}.
    :param params: Tetrad Parameters object returned by \code{\link{data.transform_pandasdf_to_tetrad_boxdataset}} (and configured by \code{\link{bootstrapping.bootstrapping}} and \code{\link{score.use_sem_bic}}).
    :param score: Tetrad score object returned by \code{\link{score.use_sem_bic}}.
    :param knowledge: Tetrad Knowledge object returned by \code{\link{data.transform_pandasdf_to_tetrad_boxdataset}} or replaced via \code{\link{knowledge.parse_knowledge_txt}}.
    :param num_starts: Number of random starts
    :param use_bes: TRUE if the final BES (Backward Equivalence Search) step is
        used from the GES (Greedy Equivalence Search) algorithm.
        This step is needed for correctness but for large models,
        since usually nearly all edges are oriented in the CPDAG,
        it is heuristically not needed.
    :param time_lag: This creates a time-series model automatically with a certain
        number of lags. Defaults to zero.
    :param use_data_order: TRUE just in case data variable order should be used for the first initial permutation.
    :param output_cpdag: Whether to output CPDAG
    :param num_threads: The number of threads (>= 1) to use for the search. Defaults to 5.
    :returns: dict with 'graph' (Java graph object) and 'bootstrap_graphs'

    :references: Dimitris Margaritis and Sebastian Thrun. Bayesian network induction via
        local neighborhoods. Advances in neural information processing systems, 12, 1999.
    :references: G., & Uhler, C. (2018). Learning directed acyclic graph models based on
        sparsest permutations. Stat, 7(1), e183.
    :references: Solus, L., Wang, Y., Matejovicova, L., & Uhler, C. (2017). Consistency
        guarantees for permutation-based causal inference algorithms. arXiv preprint arXiv:1702.03530.
    :references: Lam, W. Y., Andrews, B., & Ramsey, J. (2022, August). Greedy relaxations of
        the sparsest permutation algorithm. In Uncertainty in Artificial Intelligence (pp. 1052-1062). PMLR.
    """
    import edu.cmu.tetrad.algcomparison.algorithm.oracle.cpdag as cpdag
    from edu.cmu.tetrad.util import Params

    params.set(Params.USE_BES, use_bes)
    params.set(Params.NUM_STARTS, num_starts)
    params.set(Params.TIME_LAG, time_lag)
    params.set(Params.USE_DATA_ORDER, use_data_order)
    params.set(Params.OUTPUT_CPDAG, output_cpdag)
    params.set(Params.NUM_THREADS, num_threads)

    alg = cpdag.Boss(score)
    alg.setKnowledge(knowledge)

    graph = alg.search(data, params)
    bootstrap_graphs = alg.getBootstrapGraphs()

    return {"graph": graph, "bootstrap_graphs": bootstrap_graphs}
