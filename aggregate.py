"""Aggregates the runs of evaluate.py.

- <name>_summary.csv: per experiment, model, test set and float / INT8, the mean of every metric over
  the runs and its standard error (standard deviation / sqrt(runs)), with the costs;
- <name>_paired.csv: every pair of models of the same experiment, test set and float / INT8, compared
  seed by seed (run k of both models sees the same draw: the paired difference removes the variance
  the seeds share, which the confidence intervals of the two means keep): the mean difference
  model_b - model_a over the seeds both have, its 95 % confidence interval (Student's t), the paired
  t-test's p and its Holm correction over the pairs of the same experiment, set, kind and metric.
  detection_paired.csv also has the cost of streaming, experiment "stream_vs_window_nc": each causal
  Conformer in streaming (detection_stream, model_b) against its non-causal variant on 3 s windows
  (detection_window, model_a = <model>_nc), its best potential; Holm over these pairs.

    uv run aggregate.py         # -> results/{classification,detection}_{summary,paired}.csv
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

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


def holm(p: pd.Series) -> pd.Series:
    """Holm's step-down correction of a family of p-values (NaN left out)."""
    q = p.dropna().sort_values()
    m = len(q)
    adjusted = np.maximum.accumulate(np.minimum(1, (m - np.arange(m)) * q.to_numpy()))
    return pd.Series(adjusted, index=q.index).reindex(p.index)


def stream_vs_window_nc(runs: pd.DataFrame) -> pd.DataFrame:
    """The detection runs recast as one pseudo-experiment "stream_vs_window_nc": every causal Conformer of
    detection_stream and the non-causal variant (<model>_nc) of detection_window, for paired()."""
    stream = runs[runs.experiment == "detection_stream"]
    window = runs[(runs.experiment == "detection_window") & runs.model.isin(stream.model + "_nc")]
    return pd.concat([stream, window]).assign(experiment="stream_vs_window_nc")


def paired(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in runs.columns if c not in KEYS + ["seed"]]
    rows = []
    for (experiment, split, kind), g in runs.groupby(["experiment", "split", "kind"], sort=False):
        if experiment == "stream_vs_window_nc":   # only each model against its own non-causal variant
            pairs = [(f"{m}_nc", m) for m in g.model.unique() if f"{m}_nc" in set(g.model)]
        else:
            pairs = None
        for metric in metrics:
            table = g.pivot(index="seed", columns="model", values=metric)
            for a, b in pairs if pairs is not None else itertools.combinations(table.columns, 2):
                if a not in table or b not in table:
                    continue
                d = (table[b] - table[a]).dropna()
                n = len(d)
                if n < 2:
                    continue
                sd = d.std(ddof=1)
                rows.append({"experiment": experiment, "split": split, "kind": kind, "metric": metric,
                             "model_a": a, "model_b": b, "runs": n, "diff_mean": d.mean(),
                             "diff_ci95": stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n),
                             "p": stats.ttest_1samp(d, 0).pvalue if sd > 0 else np.nan})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = out.groupby(["experiment", "split", "kind", "metric"], sort=False)["p"].transform(holm)
    return out


def main():
    costs = pd.read_csv(RESULTS / "costs.csv") if (RESULTS / "costs.csv").exists() else None
    for name in ("classification", "detection"):
        path = RESULTS / f"{name}_runs.csv"
        if path.exists():
            summary = summarize(pd.read_csv(path), costs)
            summary.to_csv(RESULTS / f"{name}_summary.csv", index=False, float_format="%.6g")
            print(summary.to_string(index=False, max_cols=12))
            runs = pd.read_csv(path)
            if name == "detection":
                runs = pd.concat([runs, stream_vs_window_nc(runs)], ignore_index=True)
            pairs = paired(runs)
            pairs.to_csv(RESULTS / f"{name}_paired.csv", index=False, float_format="%.6g")
            print(f"{name}_paired.csv: {len(pairs)} comparisons")


if __name__ == "__main__":
    main()
