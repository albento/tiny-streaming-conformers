"""Figures of the paper, from the runs and evaluate.py's curves, into results/figures/.

    uv run plot.py                                                  # every figure
    uv run plot.py --ci                                             # with the 95 % confidence bands
    uv run plot.py labels SSW_102_20170415_161339Z --start 3153 --end 3171    # one stretch of a recording

Classification: precision / recall on the test clips, every species pooled, one figure for float and one
for INT8, whole and zoomed on precision and recall >= 0.8 (*_zoom.png).
Detection: the segment ROC and the boxes captured against the time recorded by the simulated triggered
recorder. Every curve is the mean over the runs (each run's curve on a common grid). --ci adds its 95 %
confidence interval as a band (Student's t, n - 1 degrees of freedom; none with one run): a check of the
runs' spread, not a comparison of models, whose runs share their seeds (aggregate.py's *_paired.csv
compares them seed by seed). The models are grouped as the comparisons of the paper:
the causal ViTs with the baselines, and the non-causal ViTs with the baselines (the same
regime: 3 s windows); for detection_stream, the causal ViTs alone (the device's regime).
"""
import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from paths import DATA, ROOT, RUNS  # noqa: E402

FIGURES = ROOT / "results" / "figures"
BASELINES = ("mel_polychirp_x2", "mel_polychirp_x3", "wrennet")
GRID = np.linspace(0, 1, 201)
CI = False  # --ci: the 95 % confidence bands
ZOOM = (0.8, 1.002)  # precision and recall shown by the zoomed classification figures


def mean_ci(ax, x, curves, label):
    """The mean of the runs' curves (one per row, on x) and, with --ci, its 95 % confidence interval."""
    from scipy import stats
    c = np.asarray(curves)
    m = c.mean(axis=0)
    line, = ax.plot(x, m, label=label)
    if CI and len(c) > 1:
        h = stats.t.ppf(0.975, len(c) - 1) * c.std(axis=0, ddof=1) / np.sqrt(len(c))
        ax.fill_between(x, m - h, m + h, color=line.get_color(), alpha=0.2, lw=0)


def groups(experiment: str) -> dict[str, list[str]]:
    models = sorted(p.name for p in (RUNS / experiment).glob("*") if p.is_dir())
    causal = [m for m in models if m.startswith("vit_") and not m.endswith("_nc")]
    noncausal = [m for m in models if m.endswith("_nc")]
    base = [m for m in models if m in BASELINES]
    if experiment == "detection_stream":
        return {"causal": causal}
    return {"causal": causal + base, "noncausal": noncausal + base}


def run_dirs(experiment: str, model: str) -> list[Path]:
    return sorted(d for d in (RUNS / experiment / model).glob("seed_*") if (d / "train.json").exists())


def classification():
    """classification_pr_<group>_<float|int8>.png (the whole curves) and ..._zoom.png (ZOOM): one figure each."""
    from sklearn.metrics import precision_recall_curve
    recall = np.linspace(0, 1, 1001)  # finer than GRID: the zoom covers a fifth of it
    kinds = (("float", "float"), ("INT8", "quant"))
    for group, models in groups("classification").items():
        curves = {}
        for kind, key in kinds:
            for m in models:
                for d in run_dirs("classification", m):
                    s = np.load(d / ("scores.npz" if (d / "scores.npz").exists() else "results.npz"))
                    if f"y_score_{key}_test" in s:
                        p, r, _ = precision_recall_curve(s["y_true_test"].ravel(), s[f"y_score_{key}_test"].ravel())
                        curves.setdefault((kind, m), []).append(np.interp(recall, r[::-1], p[::-1]))  # recall increasing
        for kind, _ in kinds:
            shown = [m for m in models if (kind, m) in curves]
            if not shown:  # the non-causal ViTs have no INT8 export
                continue
            for suffix, lim in (("", (0, 1.01)), ("_zoom", ZOOM)):
                fig, ax = plt.subplots(figsize=(6.5, 5.5))
                for m in shown:
                    mean_ci(ax, recall, curves[kind, m], f"{m} ({len(curves[kind, m])} runs)")
                ax.set(title=f"classification, test, {kind}", xlabel="recall", ylabel="precision", xlim=lim[:1] + (1,), ylim=lim)
                ax.grid(alpha=0.3)
                ax.legend(fontsize=7, loc="lower left")
                fig.tight_layout()
                fig.savefig(FIGURES / f"classification_pr_{group}_{kind.lower()}{suffix}.png", dpi=120)
                plt.close(fig)


def detection():
    for experiment in ("detection_stream", "detection_window"):
        kind = "int8" if experiment == "detection_stream" else "float"
        for group, models in groups(experiment).items():
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            for col, split in enumerate(("ssw_test", "sne_test")):
                for m in models:
                    curves = [np.load(d / "curves.npz") for d in run_dirs(experiment, m) if (d / "curves.npz").exists()]
                    curves = [c for c in curves if f"{split}/{kind}/roc_fpr" in c]
                    if not curves:
                        continue
                    tpr = [100 * np.interp(GRID, c[f"{split}/{kind}/roc_fpr"], c[f"{split}/{kind}/roc_tpr"]) for c in curves]
                    boxes = [100 * np.interp(GRID, c[f"{split}/{kind}/time_recorded"], c[f"{split}/{kind}/boxes_captured"]) for c in curves]
                    mean_ci(axes[0, col], 100 * GRID, tpr, f"{m} ({len(curves)} runs)")
                    mean_ci(axes[1, col], 100 * GRID, boxes, m)
                axes[0, col].set(title=f"{split}: segment ROC ({kind})", xlabel="bird-free segments flagged (%)", ylabel="segments with a box flagged (%)")
                axes[1, col].set(title=f"{split}: triggered recording ({kind})", xlabel="time recorded (%)", ylabel="annotation boxes captured (%)")
                for ax in axes[:, col]:
                    ax.plot([0, 100], [0, 100], color="0.8", lw=0.8, ls="--", zorder=0)
                    ax.grid(alpha=0.3)
                    ax.legend(fontsize=7, loc="lower right")
            fig.tight_layout()
            fig.savefig(FIGURES / f"{experiment}_{group}.png", dpi=120)
            plt.close(fig)


def labels(recording: str, start: float, end: float):
    """Spectrogram of a stretch, the annotation boxes, and every 64 ms buffer coloured by its label."""
    coll = recording.split("_")[0].lower()
    audio = np.load(DATA / coll / "audio16k" / f"{recording}.npy", mmap_mode="r")
    lab = np.load(DATA / coll / "labels.npz")[recording]
    b0, b1 = int(start * 16000 / 1024), int(np.ceil(end * 16000 / 1024))
    x = np.asarray(audio[b0 * 1024:b1 * 1024], np.float64)
    frames = np.lib.stride_tricks.sliding_window_view(x, 512)[::160] * np.hanning(513)[:-1]
    db = 10 * np.log10(np.abs(np.fft.rfft(frames, axis=1)) ** 2 + 1e-3)
    t0, t1 = b0 * 0.064, b1 * 0.064
    fig, (ax, strip) = plt.subplots(2, 1, figsize=(max(10, min(24, (t1 - t0) * 0.4)), 5), sharex=True, gridspec_kw={"height_ratios": [6, 1]})
    ax.imshow(db.T, origin="lower", aspect="auto", cmap="magma", extent=[t0, t0 + len(db) * 0.01, 0, 8000],
              vmin=np.percentile(db, 5), vmax=np.percentile(db, 99.5))
    with open(DATA / coll / "annotations.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            s, e = float(r["Start Time (s)"]), float(r["End Time (s)"])
            if Path(r["Filename"]).stem == recording and e > t0 and s < t1:
                lo, hi = float(r["Low Freq (Hz)"]), min(float(r["High Freq (Hz)"]), 8000)
                ax.add_patch(plt.Rectangle((s, lo), e - s, hi - lo, fill=False, ec="cyan", lw=1))
                ax.text(max(s, t0), hi, r["Species eBird Code"], color="cyan", fontsize=7, va="top", clip_on=True)
    ax.set(ylabel="Hz", title=f"{recording}, {t0:.2f}-{t1:.2f} s (cyan: annotation boxes)")
    colours = {0: "white", 1: "tab:green", 255: "0.6"}
    for b, v in enumerate(lab[b0:b1]):
        strip.axvspan((b0 + b) * 0.064, (b0 + b + 1) * 0.064, color=colours[int(v)], lw=0)
    strip.set(yticks=[], xlim=(t0, t1), xlabel="s", ylabel="label")
    strip.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=c, ec="k", lw=0.5) for c in colours.values()],
                 labels=["no bird", "bird", "uncertain"], loc="upper right", fontsize=7, ncol=3)
    fig.tight_layout()
    out = FIGURES / f"labels_{recording}_{start:g}-{end:g}s.png"
    fig.savefig(out, dpi=100)
    print(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("what", nargs="?", choices=("labels",), help="labels: one stretch of a recording")
    parser.add_argument("recording", nargs="?")
    parser.add_argument("--start", type=float)
    parser.add_argument("--end", type=float)
    parser.add_argument("--ci", action="store_true", help="the 95 %% confidence bands of the mean curves")
    args = parser.parse_args()
    global CI
    CI = args.ci
    FIGURES.mkdir(parents=True, exist_ok=True)
    if args.what == "labels":
        return labels(args.recording, args.start, args.end)
    if (RUNS / "classification").exists():
        classification()
    detection()


if __name__ == "__main__":
    main()
