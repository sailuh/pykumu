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
