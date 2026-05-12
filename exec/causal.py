#!/usr/bin/env python3

# Pykumu - https://github.com/sailuh/pykumu
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Command-line access to the pykumu api package for causal analysis.

USAGE:
  causal.py algorithm run-fges help
  causal.py algorithm run-fges <jar_path> <data_path> <output_path> [options]
  causal.py algorithm run-boss help
  causal.py algorithm run-boss <jar_path> <data_path> <output_path> [options]
  causal.py graph parse help
  causal.py graph parse <graph_path> <output_dir>
  causal.py graph convert-gui help
  causal.py graph convert-gui <graph_path> <output_path>
  causal.py (-h | --help)
  causal.py --version

DESCRIPTION:
  Provides command-line access to the pykumu api package for causal
  analysis using the Tetrad library. Please see pykumu's README.md
  for instructions on configuration.

OPTIONS:
  -h --help                                Show this screen.
  --version                                Show version.

  score.use_sem_bic:
  --penalty-discount=<pd>                  Penalty discount [default: 2].
  --sem-bic-structure-prior=<sp>           Structure prior [default: 0].
  --sem-bic-rule=<rule>                    BIC rule [default: 1].
  --singularity-lambda=<sl>               Singularity lambda [default: 0.0].

  bootstrapping.set_bootstrapping:
  --number-resampling=<nr>                 Bootstrap iterations, 0 = off [default: 0].
  --percent-resample-size=<prs>            Percent resample size [default: 100].
  --no-add-original-dataset                Do not add original dataset.
  --no-resampling-with-replacement         Sample without replacement.
  --resampling-ensemble=<re>               1=Preserved, 2=Highest, 3=Majority [default: 1].
  --seed=<seed>                            Random seed, -1 = off [default: -1].

  knowledge.load_knowledge:
  --knowledge=<path>                       Path to a Tetrad knowledge file.

  algorithm:
  --num-threads=<nt>                       Number of threads for the search, >= 1 [default: 1].

  algorithm.run_fges:
  --symmetric-first-step                   Score both X->Y and Y->X in first step.
  --max-degree=<md>                        Maximum graph degree, -1 = unlimited [default: -1].
  --parallelized                           Parallelize the search.
  --faithfulness-assumed                    Assume one-edge faithfulness.

  algorithm.run_boss:
  --num-starts=<ns>                        Number of random starts [default: 1].
  --use-bes                                Use final BES step.
  --time-lag=<tl>                          Time-series lag [default: 0].
  --no-use-data-order                      Do not use data variable order for first permutation.
  --no-output-cpdag                        Do not output CPDAG.
"""


import os
import sys

import pandas as pd
from docopt import docopt

# Make the repo root importable so `from api import ...` works when running
# this script directly (e.g. `python exec/causal.py ...`) without `pip install`.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from api import tetrad, data, score, bootstrapping, knowledge, algorithm, graph


def main():
    arguments = docopt(__doc__, version='Pykumu 0.0.0.9000')

    if arguments["algorithm"] and arguments["run-fges"] and arguments["help"]:
        print("Runs the FGES causal search algorithm using algorithm.run_fges().")
        print("Calls: tetrad.start(), data.load_continuous(), score.use_sem_bic(),")
        print("       bootstrapping.set_bootstrapping(), knowledge.load_knowledge(),")
        print("       algorithm.run_fges(), graph.get_json().")

    elif arguments["algorithm"] and arguments["run-fges"]:

        jar_path = arguments["<jar_path>"]
        data_path = arguments["<data_path>"]
        output_path = arguments["<output_path>"]

        # tetrad.start()
        print(f"tetrad.start({jar_path!r})")
        tetrad.start(jar_path)

        # data.load_continuous()
        print(f"data.load_continuous(pd.read_csv({data_path!r}))")
        df = pd.read_csv(data_path)
        df = df.astype(float)
        print(f"  -> {df.shape[0]} rows x {df.shape[1]} columns")
        state = data.load_continuous(df)

        # score.use_sem_bic()
        penalty_discount = float(arguments["--penalty-discount"])
        print(f"score.use_sem_bic(penalty_discount={penalty_discount})")
        sem_bic = score.use_sem_bic(
            state["params"],
            penalty_discount=penalty_discount,
            sem_bic_structure_prior=float(arguments["--sem-bic-structure-prior"]),
            sem_bic_rule=int(arguments["--sem-bic-rule"]),
            singularity_lambda=float(arguments["--singularity-lambda"]),
        )

        # bootstrapping.set_bootstrapping()
        number_resampling = int(arguments["--number-resampling"])
        if number_resampling > 0:
            print(f"bootstrapping.set_bootstrapping(number_resampling={number_resampling})")
            bootstrapping.set_bootstrapping(
                state["params"],
                number_resampling=number_resampling,
                percent_resample_size=int(arguments["--percent-resample-size"]),
                add_original_dataset=not arguments["--no-add-original-dataset"],
                resampling_with_replacement=not arguments["--no-resampling-with-replacement"],
                resampling_ensemble=int(arguments["--resampling-ensemble"]),
                seed=int(arguments["--seed"]),
            )

        # knowledge.load_knowledge()
        if arguments["--knowledge"]:
            knowledge_path = arguments["--knowledge"]
            print(f"knowledge.load_knowledge({knowledge_path!r})")
            state["knowledge"] = knowledge.load_knowledge(knowledge_path)

        # algorithm.run_fges()
        print("algorithm.run_fges()")
        result = algorithm.run_fges(
            state["data"], state["params"], sem_bic, state["knowledge"],
            symmetric_first_step=arguments["--symmetric-first-step"],
            max_degree=int(arguments["--max-degree"]),
            parallelized=arguments["--parallelized"],
            faithfulness_assumed=arguments["--faithfulness-assumed"],
            num_threads=int(arguments["--num-threads"]),
        )

        # graph.get_json()
        output_dir = os.path.dirname(os.path.abspath(output_path))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        json_str = graph.get_json(result["graph"])
        with open(output_path, "w") as f:
            f.write(json_str)
        print(f"graph.get_json() -> {output_path}")

    elif arguments["algorithm"] and arguments["run-boss"] and arguments["help"]:
        print("Runs the BOSS causal search algorithm using algorithm.run_boss().")
        print("Calls: tetrad.start(), data.load_continuous(), score.use_sem_bic(),")
        print("       bootstrapping.set_bootstrapping(), knowledge.load_knowledge(),")
        print("       algorithm.run_boss(), graph.get_json().")

    elif arguments["algorithm"] and arguments["run-boss"]:

        jar_path = arguments["<jar_path>"]
        data_path = arguments["<data_path>"]
        output_path = arguments["<output_path>"]

        # tetrad.start()
        print(f"tetrad.start({jar_path!r})")
        tetrad.start(jar_path)

        # data.load_continuous()
        print(f"data.load_continuous(pd.read_csv({data_path!r}))")
        df = pd.read_csv(data_path)
        df = df.astype(float)
        print(f"  -> {df.shape[0]} rows x {df.shape[1]} columns")
        state = data.load_continuous(df)

        # score.use_sem_bic()
        penalty_discount = float(arguments["--penalty-discount"])
        print(f"score.use_sem_bic(penalty_discount={penalty_discount})")
        sem_bic = score.use_sem_bic(
            state["params"],
            penalty_discount=penalty_discount,
            sem_bic_structure_prior=float(arguments["--sem-bic-structure-prior"]),
            sem_bic_rule=int(arguments["--sem-bic-rule"]),
            singularity_lambda=float(arguments["--singularity-lambda"]),
        )

        # bootstrapping.set_bootstrapping()
        number_resampling = int(arguments["--number-resampling"])
        if number_resampling > 0:
            print(f"bootstrapping.set_bootstrapping(number_resampling={number_resampling})")
            bootstrapping.set_bootstrapping(
                state["params"],
                number_resampling=number_resampling,
                percent_resample_size=int(arguments["--percent-resample-size"]),
                add_original_dataset=not arguments["--no-add-original-dataset"],
                resampling_with_replacement=not arguments["--no-resampling-with-replacement"],
                resampling_ensemble=int(arguments["--resampling-ensemble"]),
                seed=int(arguments["--seed"]),
            )

        # knowledge.load_knowledge()
        if arguments["--knowledge"]:
            knowledge_path = arguments["--knowledge"]
            print(f"knowledge.load_knowledge({knowledge_path!r})")
            state["knowledge"] = knowledge.load_knowledge(knowledge_path)

        # algorithm.run_boss()
        print("algorithm.run_boss()")
        result = algorithm.run_boss(
            state["data"], state["params"], sem_bic, state["knowledge"],
            num_starts=int(arguments["--num-starts"]),
            use_bes=arguments["--use-bes"],
            time_lag=int(arguments["--time-lag"]),
            use_data_order=not arguments["--no-use-data-order"],
            output_cpdag=not arguments["--no-output-cpdag"],
            num_threads=int(arguments["--num-threads"]),
        )

        # graph.get_json()
        output_dir = os.path.dirname(os.path.abspath(output_path))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        json_str = graph.get_json(result["graph"])
        with open(output_path, "w") as f:
            f.write(json_str)
        print(f"graph.get_json() -> {output_path}")

    elif arguments["graph"] and arguments["convert-gui"] and arguments["help"]:
        print("Converts a Tetrad JSON graph to Tetrad GUI format using graph.convert_to_tetrad_gui_format().")

    elif arguments["graph"] and arguments["convert-gui"]:

        graph_path = arguments["<graph_path>"]
        output_path = arguments["<output_path>"]

        # graph.convert_to_tetrad_gui_format()
        print(f"graph.convert_to_tetrad_gui_format({graph_path!r})")
        gui_json = graph.convert_to_tetrad_gui_format(graph_path)

        output_dir = os.path.dirname(os.path.abspath(output_path))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(gui_json)
        print(f"  -> {output_path}")

    elif arguments["graph"] and arguments["parse"] and arguments["help"]:
        print("Parses a Tetrad JSON graph file into CSV tables using graph.parse_graph().")

    elif arguments["graph"] and arguments["parse"]:

        graph_path = arguments["<graph_path>"]
        output_dir = arguments["<output_dir>"]

        # graph.parse_graph()
        print(f"graph.parse_graph({graph_path!r})")
        parsed = graph.parse_graph(graph_path)

        os.makedirs(output_dir, exist_ok=True)

        for name, df in parsed.items():
            out_path = os.path.join(output_dir, f"{name}.csv")
            df.to_csv(out_path, index=False)
            print(f"  {name}: {len(df)} rows -> {out_path}")

    elif arguments["--help"]:
        print(__doc__)
    elif arguments["--version"]:
        print("Pykumu 0.0.0.9000")
    else:
        print("Invalid command or arguments. Use --help for usage information.")
        sys.exit(1)


if __name__ == "__main__":
    main()
