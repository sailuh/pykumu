# Pykumu - https://github.com/sailuh/pykumu
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Graph serialization and parsing utilities.

This module provides functions to convert Tetrad graph objects
into serializable formats (e.g., JSON) and parse Tetrad JSON graph
files into tabular DataFrames.
"""

try:
    import edu.cmu.tetrad.algcomparison.algorithm.oracle.cpdag as cpdag
    from edu.cmu.tetrad.util import Params
except ImportError:
    pass  

import json

import pandas as pd
def transform_graph_java_to_graph_json(graph):
    """Convert a Tetrad graph object to a JSON string.

    :param graph: Tetrad Java graph object returned by [algorithm_fges](algorithm.html#algorithm_fges) or [algorithm.algorithm_boss](algorithm.html#algorithm_boss) (the `'graph'` key of their result dict).
    :returns: JSON string representation of the graph.
    """
    import edu.cmu.tetrad.graph.GraphSaveLoadUtils as gp

    return str(gp.graphToJson(graph))


def convert_to_tetrad_gui_format(graph_json_str):
    """Convert a Tetrad JSON graph string to the format Tetrad GUI can load.

    The newer Tetrad library serializes nodeType as a string (e.g., "MEASURED"),
    while Tetrad GUI and causal-cmd expect it as an object (e.g., {"ordinal": 0}).
    This function converts between the two formats.

    :param graph_json_str: JSON string or file path to a graph JSON
    :returns: JSON string in Tetrad GUI-compatible format
    """
    NODE_TYPE_MAP = {
        "MEASURED": {"ordinal": 0},
        "LATENT": {"ordinal": 1},
        "ERROR": {"ordinal": 2},
    }

    if graph_json_str.endswith('.json'):
        with open(graph_json_str, 'r') as f:
            graph = json.load(f)
    else:
        graph = json.loads(graph_json_str)

    # Convert nodeType in all nodes
    def convert_node(node):
        if isinstance(node.get("nodeType"), str):
            node["nodeType"] = NODE_TYPE_MAP.get(node["nodeType"], {"ordinal": 0})
        # Remove extra fields Tetrad GUI doesn't expect
        node.pop("rank", None)
        node.pop("selectionBias", None)
        return node

    for node in graph.get("nodes", []):
        convert_node(node)

    # Convert nodes inside namesHash
    for name, node in graph.get("namesHash", {}).items():
        convert_node(node)

    # Convert nodes inside edges
    for edge in graph.get("edgesSet", []):
        convert_node(edge.get("node1", {}))
        convert_node(edge.get("node2", {}))

    # Convert nodes inside edgeLists
    for name, edges in graph.get("edgeLists", {}).items():
        for edge in edges:
            convert_node(edge.get("node1", {}))
            convert_node(edge.get("node2", {}))

    # Remove extra top-level keys Tetrad GUI doesn't expect
    for key in ["ancestorCache", "potentiallyDirectedPathCache", "ancillaryGraphs", "parentsHash"]:
        graph.pop(key, None)

    return json.dumps(graph, indent=2)


def parse_graph_json(graph_filepath):
    """Parse a Tetrad JSON graph file into nodes, edgeset, and edge_type_probabilities DataFrames.

    In Tetrad, causal algorithms can output a graph as a result of a single run,
    or multiple runs. When utilizing a single run, the "nodes" and "edgeset" will
    contain the expected graph as an edgelist table and its nodes.

    A graph output resulting from multiple runs can be obtained by using either
    bootstrap or the restart flags. The bootstrap command, as the name implies,
    performs multiple causal searches over samples of the full dataset. The
    restart command, however, performs causal searches on the full dataset every
    time with random initialization (requires random initialization flag).

    If multiple runs are used, then the graph data generated in the JSON will
    contain an additional field, "edgeTypeProbabilities", for every edge in
    "edgeset". The "edgeTypeProbabilities" counts for a given edge in "edgeset",
    the number of edges in a given direction and their properties, and/or the
    absence of them of every run. The resulting "edgeset" is thus the ensemble
    edge derived from the edgeset. For example, suppose across 1000 runs for
    node1 and node2 we obtain 4 types of edges:

      * ta:  properties = (pd, pl), probability: ~0.48
      * at:  properties = (dd, nl), probability: ~0.37
      * tt:  properties = (),       probability: ~0.13
      * nil: properties = (),       probability: ~0.01

    In such edgeTypeProbabilities for node1 and node2, the reported edgeset for
    node1 and node2 is ta, the properties are (pd, pl) and the probability
    0.48 + 0.37 + 0.13 ~= 0.98. Observe the final reported property for the
    node1, node2 pair is thus the highest probability of the edgeset (assuming
    ensemble preserved), however the probability is the sum of the individual
    probabilities (except for the nil case, which were causal search runs which
    resulted in no edges being formed).

    :param graph_filepath: File path to a Tetrad JSON graph file.
    :returns: dict with 'nodes', 'edgeset', and 'edge_type_probabilities' DataFrames.
    """
    with open(graph_filepath, 'r') as f:
        graph_json = json.load(f)

    nodes = pd.DataFrame({"node_name": [n["name"] for n in graph_json["nodes"]]})

    edgeset_rows = []
    etp_rows = []

    for edge in graph_json.get("edgesSet", []):
        node1_name = edge["node1"]["name"]
        node2_name = edge["node2"]["name"]
        endpoint1 = edge.get("endpoint1")
        endpoint2 = edge.get("endpoint2")
        bold = edge.get("bold")
        highlighted = edge.get("highlighted")
        properties = ";".join(edge.get("properties", [])) or None
        probability = edge.get("probability")

        edgeset_rows.append({
            "node1_name": node1_name, "node2_name": node2_name,
            "endpoint1": endpoint1, "endpoint2": endpoint2,
            "bold": bold, "highlighted": highlighted,
            "properties": properties, "probability": probability
        })

        for etp in edge.get("edgeTypeProbabilities", []):
            etp_props = ";".join(etp.get("properties", [])) or None
            etp_rows.append({
                "node1_name": node1_name, "node2_name": node2_name,
                "edge_type": etp.get("edgeType"),
                "properties": etp_props,
                "probability": etp.get("probability")
            })

    edgeset = pd.DataFrame(edgeset_rows)
    edge_type_probabilities = pd.DataFrame(etp_rows)

    return {"nodes": nodes, "edgeset": edgeset, "edge_type_probabilities": edge_type_probabilities}


def find_cycles(g):
    """Enumerate all simple directed cycles in an igraph graph.

    Walks each vertex with positive in-degree and follows outgoing simple
    paths back to the source, recording cycles whose smallest vertex index
    is the start vertex (so each cycle is reported exactly once).

    :param g: A directed `igraph.Graph` whose vertices have a `name` attribute.
    :returns: List of cycles; each cycle is a list of vertex names in traversal order.
    :references: https://stackoverflow.com/a/55094319/1260232
    """
    cycles = []
    for v1 in g.vs:
        if g.degree(v1, mode="in") == 0:
            continue
        for v2 in [n for n in g.neighbors(v1, mode="out") if n > v1.index]:
            for path in g.get_all_simple_paths(v2, v1.index, mode="out"):
                full = [v1.index] + path
                if len(full) > 3 and min(full) == full[0]:
                    cycles.append([g.vs[i]["name"] for i in full])
    return cycles


def find_bidirectional_edges(edge_type_probabilities, min_probability=0.1):
    """Find edges whose direction is split across bootstrap runs.

    :param edge_type_probabilities: The `'edge_type_probabilities'` DataFrame returned by
        [parse_graph_json](#parse_graph_json).
    :param min_probability: Minimum share of runs each direction must reach to be reported.
    :returns: DataFrame with `node1_name`, `node2_name`, `forward_probability`
        (node1 -> node2, edge type `ta`) and `backward_probability` (node2 -> node1, edge type `at`).
    """
    pivot = (edge_type_probabilities
             .pivot_table(index=["node1_name", "node2_name"], columns="edge_type",
                          values="probability", aggfunc="sum", fill_value=0.0)
             .reset_index())
    for col in ["ta", "at"]:
        if col not in pivot.columns:
            pivot[col] = 0.0
    result = pivot.rename(columns={"ta": "forward_probability", "at": "backward_probability"})
    both = ((result["forward_probability"] >= min_probability)
            & (result["backward_probability"] >= min_probability))
    columns = ["node1_name", "node2_name", "forward_probability", "backward_probability"]
    return result.loc[both, columns].reset_index(drop=True)


def find_cycles_in_edgeset(edgeset):
    """Find directed cycles in a causal graph edge set.

    Only directed edges are used; undirected (TAIL-TAIL) edges are skipped.

    :param edgeset: An edge set DataFrame like the `'edgeset'` returned by
        [parse_graph_json](#parse_graph_json).
    :returns: List of cycles; each cycle is a list of node names, starting and
        ending with the same node, e.g. `[["A", "B", "C", "A"]]`.
    """
    import igraph as ig
    directed = []
    for _, row in edgeset.iterrows():
        if row["endpoint1"] == "TAIL" and row["endpoint2"] == "ARROW":
            directed.append((row["node1_name"], row["node2_name"]))
        elif row["endpoint1"] == "ARROW" and row["endpoint2"] == "TAIL":
            directed.append((row["node2_name"], row["node1_name"]))
    if not directed:
        return []
    g = ig.Graph.TupleList(directed, directed=True)
    return find_cycles(g)

def markov_check(data, params, test, graph, knowledge=None, fraction_resample=1.0,
                 find_smallest_subset=False, condition_set_type="ORDERED_LOCAL_MARKOV",
                 parallelized=True):
    """Check whether a causal graph fits the data using Tetrad's Markov checker.

    :param data: Tetrad data object from [transform_pandasdf_to_tetrad_boxdataset](data.html#transform_pandasdf_to_tetrad_boxdataset).
    :param params: Tetrad Parameters object from the same function.
    :param test: Independence test from [use_fisher_z](independence.html#use_fisher_z).
    :param graph: Causal graph from [algorithm_fges](algorithm.html#algorithm_fges) or [algorithm_boss](algorithm.html#algorithm_boss).
    :param knowledge: Tetrad Knowledge (tiers), the same one used for the search. None = no rules.
    :param fraction_resample: Part of the data to use, from 0 to 1. 1.0 = all the data.
    :param find_smallest_subset: True to try smaller conditioning sets for each test.
    :param condition_set_type: Name of the Tetrad ConditioningSetType, e.g. `"ORDERED_LOCAL_MARKOV"` or `"LOCAL_MARKOV"`.
    :param parallelized: True to run the tests in parallel.
    :returns: dict with `'summary'` (summary statistics) and `'results'` (one row per test).
        `summary['fraction_dependent_independent']` is the share of the graph's
        predictions the data contradicts (lower is better).
    """
    import edu.cmu.tetrad.search as ts
    set_type = getattr(ts.ConditioningSetType, condition_set_type)  # text -> Tetrad option
    mc = ts.MarkovCheck(graph, test.getTest(data, params), set_type)
    mc.setParallelized(parallelized)
    mc.setFractionResample(fraction_resample)          # 1.0 = use all the data
    mc.setFindSmallestSubset(find_smallest_subset)     # py-tetrad calls this removeExtraneous
    if knowledge is not None:
        mc.setKnowledge(knowledge)                     # same tiers as the search
    mc.generateAllResults()                            # run AFTER all the settings

    summary = {
        "num_tests_independent": int(mc.getNumTests(True)),
        "fraction_dependent_independent": float(mc.getFractionDependent(True)),
        "ad_p_independent": float(mc.getAndersonDarlingP(True)),
        "ks_p_independent": float(mc.getKsPValue(True)),
        "num_tests_dependent": int(mc.getNumTests(False)),
        "fraction_dependent_dependent": float(mc.getFractionDependent(False)),
    }

    rows = []
    for implied_independent in (True, False):
        results = mc.getResults(implied_independent)
        for i in range(results.size()):
            r = results.get(i)
            rows.append({"fact": str(r.getFact()), "p_value": float(r.getPValue()),
                         "implied_independent": implied_independent})
    return {"summary": summary, "results": pd.DataFrame(rows)}