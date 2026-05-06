# Pykumu - https://github.com/sailuh/pykumu
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Bootstrapping configuration for causal search algorithms.

This module provides functions to configure bootstrapping (resampling)
parameters on a Tetrad Parameters object.
"""

try:
    from edu.cmu.tetrad.util import Params
except ImportError:
    pass  

def bootstrapping(params, number_resampling=0, percent_resample_size=100, add_original_dataset=True,
                      resampling_with_replacement=True, resampling_ensemble=1, seed=-1):
    """Configure bootstrapping parameters for causal search.

    :param params: Tetrad Parameters object returned by [transform_pandasdf_to_tetrad_boxdataset](data.html#transform_pandasdf_to_tetrad_boxdataset).
    :param number_resampling: The number of bootstraps/resampling iterations (min = 0)
    :param percent_resample_size: The percentage of resample size (min = 10%)
    :param add_original_dataset: Yes, if adding the original dataset as another bootstrapping
    :param resampling_with_replacement: Yes, if sampling with replacement (bootstrapping)
    :param resampling_ensemble: Ensemble method: Preserved (1), Highest (2), Majority (3).
        Use any other number to not include the flag.
    :param seed: Seed for pseudorandom number generator (-1 = off)
    """

    params.set(Params.NUMBER_RESAMPLING, number_resampling)
    params.set(Params.PERCENT_RESAMPLE_SIZE, percent_resample_size)
    params.set(Params.ADD_ORIGINAL_DATASET, add_original_dataset)
    params.set(Params.RESAMPLING_WITH_REPLACEMENT, resampling_with_replacement)
    params.set(Params.RESAMPLING_ENSEMBLE, resampling_ensemble)
    params.set(Params.SEED, seed)
