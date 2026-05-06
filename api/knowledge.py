# Pykumu - https://github.com/sailuh/pykumu
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Domain knowledge loading for causal search.

This module provides functions to load domain knowledge constraints
(forbidden/required edges, temporal tiers) from a .txt file.
"""

try:
    import edu.cmu.tetrad.data as td
    import java.io as io
except ImportError:
    pass  

def parse_knowledge_txt(path):
    """Load domain knowledge constraints from a file.

    The knowledge file uses whitespace delimiters and '#' for comments.
    It defines tiers and forbidden/required edges for causal search.

    :param path: File path to the knowledge file
    :returns: Tetrad Knowledge object
    """

    know_file = io.File(path)
    know_delim = td.DelimiterType.WHITESPACE
    return td.SimpleDataLoader.loadKnowledge(know_file, know_delim, "#")
