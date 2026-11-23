"""Figures of the paper, from the runs and evaluate.py's curves, into results/figures/ (PNG and EPS).

    uv run plot.py                                                  # every figure
    uv run plot.py --ci                                             # with the 95 % confidence bands
    uv run plot.py labels SSW_102_20170415_161339Z --start 3153 --end 3171    # one stretch of a recording

Float scores only: the comparison of the models (INT8 is evaluate.py's, to show it runs on the device).
One figure per group of models, two panels side by side: left, classification precision / recall on the
test clips, every species pooled; right, detection segment ROC on the SSW test days
(results_<group>.png; results_<group>_zoom.png zoomed on PR_ZOOM and ROC_ZOOM). Beside it, the boxes captured against the time recorded by the simulated triggered
recorder (trigger_<group>.png). Detection: our causal ViTs in streaming (detection_stream, the device's
regime), every other model on 3 s windows (detection_window). A model has the same colour everywhere;
our ViTs solid, WrenNet dash-dot, PolyChirp's models dashed.
Every curve is the mean over the
runs (each run's curve on a common grid). --ci adds its 95 %
confidence interval as a band (Student's t, n - 1 degrees of freedom; none with one run): a check of the
runs' spread, not a comparison of models, whose runs share their seeds (aggregate.py's *_paired.csv
compares them seed by seed). The models are grouped as the comparisons of the paper:
the causal ViTs with the baselines, and the non-causal ViTs with the baselines.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
import matplotlib.ticker
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from paths import DATA, ROOT, RUNS  # noqa: E402

FIGURES = ROOT / "results" / "figures"
BASELINES = ("mel_polychirp_x2", "mel_polychirp_x3", "wrennet")
GRID = np.linspace(0, 1, 1001)  # fine enough for the zooms
CI = False  # --ci: the 95 % confidence bands
PR_ZOOM = ((90, 100), (90, 100.1))  # recall and precision (%) shown by the zoomed classification panel
ROC_ZOOM = ((4, 20), (70, 95))  # false and true positive rates (%) shown by the zoomed ROC
FONT = 13  # pt: labels, ticks and titles of the paper's figures; legends FONT - 1.5 (under the panels), FONT - 3 (trigger_*)
SPLIT = "ssw_test"  # the detection test set (evaluate.py's curves)


def display_name(model: str) -> str:
    """The paper's names: vit_s24_l1 -> Conf-S2x4-L1 (_nc -> -nc), mel_polychirp_x2 -> Mel-PolyChirpx2."""
    if model == "wrennet":
        return "WrenNet"
    if model.startswith("mel_polychirp_"):
        return "Mel-PolyChirp" + model.removeprefix("mel_polychirp_")
    _, stem, layers, *nc = model.split("_")   # vit_s<2><k>_l<n>[_nc]
    return f"Conf-S{stem[1]}x{stem[2:]}-L{layers[1:]}" + "".join(f"-{x}" for x in nc)


def line_style(model: str) -> str:
    """Our ViTs solid, WrenNet dash-dot, PolyChirp's models dashed."""
    return "-." if model == "wrennet" else "--" if model.startswith("mel_polychirp") else "-"


def mean_ci(ax, x, curves, label, color, ls="-"):
    """The mean of the runs' curves (one per row, on x) and, with --ci, its 95 % confidence interval."""
    from scipy import stats
    c = np.asarray(curves)
    m = c.mean(axis=0)
    ax.plot(x, m, label=label, color=color, ls=ls)
    if CI and len(c) > 1:
        h = stats.t.ppf(0.975, len(c) - 1) * c.std(axis=0, ddof=1) / np.sqrt(len(c))
        band = 0.8 + 0.2 * np.array(matplotlib.colors.to_rgb(color))   # 20 % of the colour on white, opaque (EPS)
        ax.fill_between(x, m - h, m + h, color=band, lw=0, zorder=0.5)


def save(fig, name: str):
    """name.png and name.eps (the paper's, for dvips): no transparency, which PostScript lacks."""
    fig.savefig(FIGURES / f"{name}.png", dpi=120)
    fig.savefig(FIGURES / f"{name}.eps")


def groups() -> dict[str, list[str]]:
    """The models of each comparison, from the classification runs."""
    models = sorted(p.name for p in (RUNS / "classification").glob("*") if p.is_dir())
    base = [m for m in models if m in BASELINES]
    return {"causal": [m for m in models if m.startswith("vit_") and not m.endswith("_nc")] + base,
            "noncausal": [m for m in models if m.endswith("_nc")] + base}


def run_dirs(experiment: str, model: str) -> list[Path]:
    return sorted(d for d in (RUNS / experiment / model).glob("seed_*") if (d / "train.json").exists())


def detection_experiment(model: str) -> str:
    """Our causal ViTs in streaming, every other model on 3 s windows."""
    return "detection_stream" if model.startswith("vit_") and not model.endswith("_nc") else "detection_window"


def pr_curves(model: str) -> list[np.ndarray]:
    """Precision on GRID (recall), one row per classification run."""
    from sklearn.metrics import precision_recall_curve
    out = []
    for d in run_dirs("classification", model):
        s = np.load(d / ("scores.npz" if (d / "scores.npz").exists() else "results.npz"))
        p, r, _ = precision_recall_curve(s["y_true_test"].ravel(), s["y_score_float_test"].ravel())
        out.append(100 * np.interp(GRID, r[::-1], p[::-1]))  # %, recall increasing
    return out


def detection_curves(model: str, x: str, y: str) -> list[np.ndarray]:
    """100 * y on 100 * GRID (x), one row per detection run, on the SSW test days."""
    out = []
    for d in run_dirs(detection_experiment(model), model):
        if (d / "curves.npz").exists():
            c = np.load(d / "curves.npz")
            if f"{SPLIT}/float/{x}" in c:
                out.append(100 * np.interp(GRID, c[f"{SPLIT}/float/{x}"], c[f"{SPLIT}/float/{y}"]))
    return out


def diagonal(ax):
    ax.plot([0, 100], [0, 100], color="0.8", lw=0.8, ls="--", zorder=0)


def figures():
    """results_<group>[_zoom].png (classification PR | detection ROC) and trigger_<group>.png."""
    plt.rcParams.update({"font.size": FONT, "axes.titlesize": FONT, "axes.labelsize": FONT})
    for group, models in groups().items():
        colour = {m: f"C{i}" for i, m in enumerate(models)}
        pr = {m: c for m in models if (c := pr_curves(m))}
        roc = {m: c for m in models if (c := detection_curves(m, "roc_fpr", "roc_tpr"))}
        trig = {m: c for m in models if (c := detection_curves(m, "time_recorded", "boxes_captured"))}
        for suffix, pr_lim, roc_lim in (("", ((0, 100), (0, 101)), ((0, 100), (0, 100.5))), ("_zoom", PR_ZOOM, ROC_ZOOM)):
            fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
            for m, c in pr.items():
                mean_ci(left, 100 * GRID, c, display_name(m), colour[m], line_style(m))
            for m, c in roc.items():
                mean_ci(right, 100 * GRID, c, display_name(m), colour[m], line_style(m))
            left.set(title="Classification precision/recall curve", xlabel="recall (%)", ylabel="precision (%)",
                     xlim=pr_lim[0], ylim=pr_lim[1])
            right.set(title="Detection ROC curve", xlabel="false positive rate (%)",
                      ylabel="true positive rate (%)", xlim=roc_lim[0], ylim=roc_lim[1])
            diagonal(right)
            right.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))   # rates in whole %
            for ax in (left, right):
                ax.grid(color="0.88")   # opaque: the 0.3 alpha of the default grid colour
            # one legend for both panels (same models, same colours), under them: it hides no curve.
            # One row when it fits the figure's width, else two, each centred: our models, then the baselines.
            # (A legend aligns its entries on a grid: the second row is a second legend, laid over the blank
            # second row of the first, which makes the constrained layout leave room for both.)
            handles = {h.get_label(): h for ax in (left, right) for h in ax.get_legend_handles_labels()[0]}
            names = [display_name(m) for m in models if display_name(m) in handles]
            ours = [n for n in names if n.startswith("Conf-")]
            style = dict(fontsize=FONT - 1.5, frameon=False, columnspacing=1.2, handlelength=2.2)
            legend = fig.legend([handles[n] for n in names], names, loc="outside lower center", ncol=len(names), **style)
            fig.canvas.draw()
            if legend.get_window_extent().width > fig.bbox.width:
                legend.remove()
                rest = [n for n in names if n not in ours]
                blank = plt.Line2D([], [], lw=0)
                legend = fig.legend([h for n in ours for h in (handles[n], blank)], [x for n in ours for x in (n, "")],
                                    loc="outside lower center", ncol=len(ours), **style)   # column by column
                fig.canvas.draw()
                bottom = legend.get_window_extent().transformed(fig.transFigure.inverted()).y0
                fig.legend([handles[n] for n in rest], rest, loc="lower center", bbox_to_anchor=(0.5, bottom),
                           borderaxespad=0, ncol=len(rest), **style)
            save(fig, f"results_{group}{suffix}")
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        for m, c in trig.items():
            mean_ci(ax, 100 * GRID, c, display_name(m), colour[m], line_style(m))
        ax.set(title="detection (SSW test days): triggered recording", xlabel="time recorded (%)",
               ylabel="annotation boxes captured (%)")
        diagonal(ax)
        ax.grid(color="0.88")   # opaque: the 0.3 alpha of the default grid colour
        ax.legend(fontsize=FONT - 3, loc="lower right", framealpha=1)
        fig.tight_layout()
        save(fig, f"trigger_{group}")
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
    figures()


if __name__ == "__main__":
    main()
