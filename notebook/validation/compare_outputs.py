#!/usr/bin/env python3
"""
compare_outputs.py
Compares the numeric outputs from the R and Python validation scripts.
Reports PASS/FAIL for each section with details on mismatches.

Usage:
    python validation/compare_outputs.py [--r-dir validation/outputs/r] [--py-dir validation/outputs/python] [--tol 1e-6]
"""

import os
import argparse
import json
import sys

import numpy as np
import pandas as pd


def compare_json(r_path, py_path, tol):
    """Compare two JSON graph files produced by Tetrad."""
    with open(r_path) as f:
        r_json = json.load(f)
    with open(py_path) as f:
        py_json = json.load(f)

    issues = []

    # Compare nodes
    r_nodes = sorted([n["name"] for n in r_json.get("nodes", [])])
    py_nodes = sorted([n["name"] for n in py_json.get("nodes", [])])
    if r_nodes != py_nodes:
        only_r = set(r_nodes) - set(py_nodes)
        only_py = set(py_nodes) - set(r_nodes)
        if only_r:
            issues.append(f"Nodes only in R: {sorted(only_r)[:5]}...")
        if only_py:
            issues.append(f"Nodes only in Python: {sorted(only_py)[:5]}...")
    else:
        issues_count = 0

    # Compare edges
    def edge_key(e):
        return (e.get("node1", {}).get("name", ""),
                e.get("node2", {}).get("name", ""))

    r_edges = {edge_key(e): e for e in r_json.get("edgesSet", [])}
    py_edges = {edge_key(e): e for e in py_json.get("edgesSet", [])}

    only_r_edges = set(r_edges.keys()) - set(py_edges.keys())
    only_py_edges = set(py_edges.keys()) - set(r_edges.keys())
    if only_r_edges:
        issues.append(f"{len(only_r_edges)} edges only in R (e.g. {list(only_r_edges)[:3]})")
    if only_py_edges:
        issues.append(f"{len(only_py_edges)} edges only in Python (e.g. {list(only_py_edges)[:3]})")

    # Compare probabilities on shared edges
    shared = set(r_edges.keys()) & set(py_edges.keys())
    prob_mismatches = []
    for key in shared:
        r_prob = r_edges[key].get("probability", 0)
        py_prob = py_edges[key].get("probability", 0)
        if abs(r_prob - py_prob) > tol:
            prob_mismatches.append(
                f"  {key}: R={r_prob}, Py={py_prob}, diff={abs(r_prob - py_prob):.2e}"
            )
    if prob_mismatches:
        issues.append(f"{len(prob_mismatches)} probability mismatches "
                      f"(showing first {min(5, len(prob_mismatches))}):")
        issues.extend(prob_mismatches[:5])

    if not issues:
        issues_summary = f"{len(r_nodes)} nodes, {len(r_edges)} edges"
    else:
        issues_summary = None

    return issues, issues_summary


def load_value(path):
    """Load a single value from a .txt file."""
    with open(path) as f:
        return f.read().strip()


def compare_values(r_val, py_val, tol):
    """Compare two string values, trying numeric comparison first."""
    try:
        r_num = float(r_val)
        py_num = float(py_val)
        if np.isnan(r_num) and np.isnan(py_num):
            return True, ""
        diff = abs(r_num - py_num)
        if diff <= tol:
            return True, ""
        return False, f"R={r_val}, Python={py_val}, diff={diff:.2e}"
    except ValueError:
        if r_val == py_val:
            return True, ""
        return False, f"R='{r_val}', Python='{py_val}'"


def compare_tables(r_path, py_path, tol):
    """Compare two CSV files cell by cell with float tolerance."""
    r_df = pd.read_csv(r_path)
    py_df = pd.read_csv(py_path)

    issues = []

    # Check dimensions
    if r_df.shape != py_df.shape:
        issues.append(f"Shape mismatch: R={r_df.shape}, Python={py_df.shape}")
        # Compare up to the smaller dimensions
        min_rows = min(r_df.shape[0], py_df.shape[0])
        min_cols = min(r_df.shape[1], py_df.shape[1])
        r_df = r_df.iloc[:min_rows, :min_cols]
        py_df = py_df.iloc[:min_rows, :min_cols]

    # Check column names
    r_cols = list(r_df.columns)
    py_cols = list(py_df.columns)
    if r_cols != py_cols:
        # Find differences
        only_r = set(r_cols) - set(py_cols)
        only_py = set(py_cols) - set(r_cols)
        if only_r:
            issues.append(f"Columns only in R: {sorted(only_r)}")
        if only_py:
            issues.append(f"Columns only in Python: {sorted(only_py)}")

    # Cell-by-cell comparison on shared columns
    shared_cols = [c for c in r_cols if c in py_cols]
    mismatch_count = 0
    mismatch_examples = []

    for col in shared_cols:
        for i in range(min(len(r_df), len(py_df))):
            r_val = r_df[col].iloc[i]
            py_val = py_df[col].iloc[i]

            # Handle NaN
            if pd.isna(r_val) and pd.isna(py_val):
                continue

            # Numeric comparison
            try:
                r_num = float(r_val)
                py_num = float(py_val)
                if abs(r_num - py_num) > tol:
                    mismatch_count += 1
                    if len(mismatch_examples) < 5:
                        mismatch_examples.append(
                            f"  row {i}, col '{col}': R={r_num}, Py={py_num}, "
                            f"diff={abs(r_num - py_num):.2e}"
                        )
            except (ValueError, TypeError):
                if str(r_val) != str(py_val):
                    mismatch_count += 1
                    if len(mismatch_examples) < 5:
                        mismatch_examples.append(
                            f"  row {i}, col '{col}': R='{r_val}', Py='{py_val}'"
                        )

    if mismatch_count > 0:
        issues.append(f"{mismatch_count} cell mismatches (showing first {len(mismatch_examples)}):")
        issues.extend(mismatch_examples)

    return issues


def main():
    parser = argparse.ArgumentParser(description="Compare R and Python validation outputs")
    parser.add_argument("--r-dir",
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kumu_outputs"),
                        help="R (kumu) output directory")
    parser.add_argument("--py-dir",
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pykumu_outputs"),
                        help="Python (pykumu) output directory")
    parser.add_argument("--tol", type=float, default=1e-6,
                        help="Floating point tolerance (default: 1e-6)")
    args = parser.parse_args()

    r_dir = args.r_dir
    py_dir = args.py_dir
    tol = args.tol

    if not os.path.isdir(r_dir):
        print(f"ERROR: R output directory not found: {r_dir}")
        sys.exit(1)
    if not os.path.isdir(py_dir):
        print(f"ERROR: Python output directory not found: {py_dir}")
        sys.exit(1)

    # Sections to compare (filename without extension, description, file type)
    # These match the filenames saved by the R and Python notebooks
    sections = [
        ("feature_engineering",                     "Feature engineering",                   "csv"),
        ("feature_renaming",                        "Feature renaming",                      "csv"),
        ("missing_data_handling",                   "Missing data handling",                 "csv"),
        ("time_lag_features",                       "1-Time lag features",                   "csv"),
        ("correlation_matrix",                      "Correlation matrix",                    "csv"),
        ("npn_transformed",                         "NPN transformation",                    "csv"),
        ("binarized_lag_dt",                        "Binarized lag table",                   "csv"),
        ("nv_lag_dt_final",                         "Final null variable table",             "csv"),
        ("null_search_graph",                       "Null search graph JSON",                "json"),
        ("null_search_nodes",                       "Null search graph nodes",               "csv"),
        ("null_search_edgeset",                     "Null search graph edgeset",             "csv"),
        ("null_search_edge_type_probabilities",     "Null search edge type probabilities",   "csv"),
        ("nv_edges",                                "Null variable edges",                   "csv"),
        ("pnef_1",                                  "1PNEF threshold",                       "txt"),
        ("domain_search_graph",                     "Domain search graph JSON",              "json"),
        ("final_search_nodes",                      "Final search graph nodes",              "csv"),
        ("final_search_edgeset",                    "Final search graph edgeset",            "csv"),
        ("final_search_edge_type_probabilities",    "Final search edge type probabilities",  "csv"),
        ("edges_1pnef",                             "Edges after 1PNEF filtering",           "csv"),
    ]

    print("=" * 70)
    print("Validation Report: R vs Python Notebook Section Outputs")
    print(f"Tolerance: {tol}")
    print("=" * 70)
    print()

    total = 0
    passed = 0
    failed = 0
    skipped = 0

    for name, description, file_type in sections:
        total += 1
        ext = {"csv": ".csv", "txt": ".txt", "json": ".json"}[file_type]
        r_path = os.path.join(r_dir, name + ext)
        py_path = os.path.join(py_dir, name + ext)

        # Check files exist
        if not os.path.exists(r_path):
            print(f"  SKIP  {description} ({name}) — R file missing")
            skipped += 1
            continue
        if not os.path.exists(py_path):
            print(f"  SKIP  {description} ({name}) — Python file missing")
            skipped += 1
            continue

        if file_type == "txt":
            r_val = load_value(r_path)
            py_val = load_value(py_path)
            match, detail = compare_values(r_val, py_val, tol)
            if match:
                print(f"  PASS  {description} ({name}) = {r_val}")
                passed += 1
            else:
                print(f"  FAIL  {description} ({name}): {detail}")
                failed += 1
        elif file_type == "json":
            issues, summary = compare_json(r_path, py_path, tol)
            if not issues:
                print(f"  PASS  {description} ({name}) [{summary}]")
                passed += 1
            else:
                print(f"  FAIL  {description} ({name}):")
                for issue in issues:
                    print(f"        {issue}")
                failed += 1
        else:
            issues = compare_tables(r_path, py_path, tol)
            if not issues:
                r_df = pd.read_csv(r_path)
                print(f"  PASS  {description} ({name}) [{r_df.shape[0]}x{r_df.shape[1]}]")
                passed += 1
            else:
                print(f"  FAIL  {description} ({name}):")
                for issue in issues:
                    print(f"        {issue}")
                failed += 1

    print()
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped out of {total}")
    if failed == 0 and skipped == 0:
        print("ALL SECTIONS MATCH")
    elif failed > 0:
        print("MISMATCHES FOUND — review the FAIL entries above")
    print("=" * 70)

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
