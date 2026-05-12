# Pykumu - https://github.com/sailuh/pykumu
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Data loading and preparation for causal analysis.

This module converts pandas DataFrames into Tetrad-compatible Java data
structures (BoxDataSet) and initializes the search state.
"""

try:
    import java.util as util
    import edu.cmu.tetrad.data as td
    from edu.cmu.tetrad.util import Parameters
except ImportError:
    pass  

from pandas import DataFrame


def transform_pandasdf_to_tetrad_boxdataset(df: DataFrame, int_as_cont=False):
    """Convert a pandas DataFrame to Tetrad data format and initialize search state.

    Columns with float dtypes are treated as continuous variables. All other
    columns are treated as discrete variables unless ``int_as_cont`` is True,
    in which case integer columns are also treated as continuous.

    :param df: pandas DataFrame to convert.
    :param int_as_cont: If True, treat integer columns as continuous rather
        than discrete.
    :returns: dict with 'data', 'params', and 'knowledge' keys.
    """

    dtypes = ["float16", "float32", "float64"]
    if int_as_cont:
        for i in range(3, 7):
            dtypes.append(f"int{2 ** i}")
            dtypes.append(f"uint{2 ** i}")

    cols = df.columns
    discrete_cols = [col for col in cols if df[col].dtypes not in dtypes]

    category_map = {
        col: {val: i for i, val in enumerate(df[col].unique())}
        for col in discrete_cols
    }

    df = df.copy()
    for col in discrete_cols:
        s = df[col].map(category_map[col])
        df[col] = s.astype("int64")

    values = df.values
    n, p = df.shape

    variables = util.ArrayList()
    for col in cols:
        if col in discrete_cols:
            categories = util.ArrayList()
            for category in category_map[col]:
                categories.add(str(category))
            variables.add(td.DiscreteVariable(str(col), categories))
        else:
            variables.add(td.ContinuousVariable(str(col)))

    if len(discrete_cols) == len(cols):
        databox = td.IntDataBox(n, p)
    elif len(discrete_cols) == 0:
        databox = td.DoubleDataBox(n, p)
    else:
        databox = td.MixedDataBox(variables, n)

    for col, var in enumerate(values.T):
        for row, val in enumerate(var):
            databox.set(row, col, val)

    data = td.BoxDataSet(databox, variables)
    params = Parameters()
    knowledge = td.Knowledge()
    return {"data": data, "params": params, "knowledge": knowledge}
