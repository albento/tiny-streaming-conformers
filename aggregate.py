"""Aggregates the runs of evaluate.py: per experiment, model, test set and float / INT8, the mean of
every metric over the runs and its standard error (standard deviation / sqrt(runs)), with the costs.

    uv run aggregate.py         # -> results/classification_summary.csv, results/detection_summary.csv
"""
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"
KEYS = ["experiment", "model", "split", "kind"]


def summarize(runs: pd.DataFrame, costs: pd.DataFrame | None) -> pd.DataFrame:
    metrics = [c for c in runs.columns if c not in KEYS + ["seed"]]
    grouped = runs.groupby(KEYS, sort=False)
    out = grouped.size().rename("runs").to_frame()
    for m in metrics:
        out[f"{m}_mean"] = grouped[m].mean()
        out[f"{m}_sem"] = grouped[m].std(ddof=1) / out["runs"] ** 0.5
    out = out.reset_index()
    return out.merge(costs, on="model", how="left") if costs is not None else out


def main():
    costs = pd.read_csv(RESULTS / "costs.csv") if (RESULTS / "costs.csv").exists() else None
    for name in ("classification", "detection"):
        path = RESULTS / f"{name}_runs.csv"
        if path.exists():
            summary = summarize(pd.read_csv(path), costs)
            summary.to_csv(RESULTS / f"{name}_summary.csv", index=False, float_format="%.6g")
            print(summary.to_string(index=False, max_cols=12))


if __name__ == "__main__":
    main()
