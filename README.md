# Pykumu

*Pykumu* is a Python port of [Kumu](https://github.com/sailuh/kumu). *Kumu* (n.) — reason, cause, goal, justification, motive, grounds, purpose, object, why.

## Overview

Pykumu is a Python package to facilitate data transformations and causal modeling. It exposes the [Tetrad](https://github.com/cmu-phil/tetrad) causal discovery library in Python via [JPype](https://github.com/jpype-project/jpype), plus a command-line interface for scripted use.

Pykumu is forked from [cmu-phil/py-tetrad](https://github.com/cmu-phil/py-tetrad).

## Installation

Pykumu has been tested on macOS.

### 1. Prerequisites

- **Install Miniconda** — [docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
- **Install JDK 21** (Amazon Corretto 21 recommended) — [aws.amazon.com/corretto/](https://aws.amazon.com/corretto/)

Verify `java -version` prints `21.x` and `JAVA_HOME` points at the JDK 21 install.

### 2. Clone and enter the repo

```bash
git clone https://github.com/sailuh/pykumu.git
cd pykumu
```

### 3. Create the conda environment and install pykumu
From repo root:

```bash
conda env create -f env.yml
conda activate pykumu
pip install -e .
```

This installs every runtime dependency (`numpy`, `pandas`, `scipy`, `JPype1`, `python-igraph`, `pyvis`, `docopt`, `jupyterlab`).

### 4. Rebuild the documentation

```bash
make docs
```

Regenerates `docs/api/` (via `pdoc`) and `docs/notebook/` (via `jupyter nbconvert`).

## Running the notebook

End-to-end causal discovery workflow lives in [notebook/issue_causal_analysis.ipynb](notebook/issue_causal_analysis.ipynb). Open it in Jupyter or VS Code, select the `pykumu` kernel, and run all cells. See the notebook's Setup section for the full step-by-step.

## Stay up-to-date

- Issues and feature requests: [github.com/sailuh/pykumu/issues](https://github.com/sailuh/pykumu/issues)
- API documentation: [phuong808.github.io/pykumu](https://phuong808.github.io/pykumu)

## Citation

Pykumu is a Python port of the R package [sailuh/kumu](https://github.com/sailuh/kumu); if you use it in research, please cite the upstream Kumu work and Tetrad per their project guidance.
