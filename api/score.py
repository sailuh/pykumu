# Pykumu - https://github.com/sailuh/pykumu
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Score configuration for causal search algorithms.

This module provides functions to configure and return scoring
functions (e.g., SEM BIC) used by Tetrad search algorithms.
"""

try:
    import edu.cmu.tetrad.algcomparison.score as score_
    from edu.cmu.tetrad.util import Params
except ImportError:
    pass  

def use_sem_bic(params, penalty_discount=2, sem_bic_structure_prior=0, sem_bic_rule=1, singularity_lambda=0.0):
    """Configure and return a SEM BIC score.

    :param params: Tetrad Parameters object returned by [transform_pandasdf_to_tetrad_boxdataset](data.html#transform_pandasdf_to_tetrad_boxdataset).
    :param penalty_discount: Penalty discount (min = 0.0)
    :param sem_bic_structure_prior: Structure Prior for SEM BIC (default 0)
    :param sem_bic_rule: BIC rule selection (1 = default)
    :param singularity_lambda: >= 0 adds lambda to matrix diagonals, < 0 uses pseudoinverse
    :returns: SemBicScore object
    """

    params.set(Params.PENALTY_DISCOUNT, penalty_discount)
    params.set(Params.SEM_BIC_STRUCTURE_PRIOR, sem_bic_structure_prior)
    params.set(Params.SEM_BIC_RULE, sem_bic_rule)
    params.set(Params.SINGULARITY_LAMBDA, singularity_lambda)
    return score_.SemBicScore()
