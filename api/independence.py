# Pykumu - https://github.com/sailuh/pykumu
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# (license header, the same one every Pykumu file starts with)

"""Independence test configuration.

This module provides functions to configure and return independence
tests, used by the Markov checker (and by test-based search algorithms).
"""

try:
    import edu.cmu.tetrad.algcomparison.independence as ind_
    from edu.cmu.tetrad.util import Params
except ImportError:
    pass


def use_fisher_z(params, alpha=0.01, singularity_lambda=0.0):
    """Configure and return a Fisher Z independence test.

    Fisher Z tests whether two continuous variables are independent,
    optionally given a set of other variables.

    :param params: Tetrad Parameters object returned by [transform_pandasdf_to_tetrad_boxdataset](data.html#transform_pandasdf_to_tetrad_boxdataset).
    :param alpha: Significance level for the test (default 0.01).
    :param singularity_lambda: >= 0 adds lambda to matrix diagonals, < 0 uses pseudoinverse
    :returns: FisherZ test object
    """

    params.set(Params.ALPHA, alpha)
    params.set(Params.SINGULARITY_LAMBDA, singularity_lambda)
    return ind_.FisherZ()