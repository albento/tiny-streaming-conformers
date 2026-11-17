"""Figures of the paper, from the runs and evaluate.py's curves, into results/figures/.

    uv run plot.py                                                  # every figure
    uv run plot.py labels SSW_102_20170415_161339Z --start 3153 --end 3171    # one stretch of a recording

Classification: precision / recall on the test clips, every species and run pooled, float and INT8.
Detection: the segment ROC (pooled over runs) and the boxes captured against the time recorded by the
simulated triggered recorder (mean over runs). The models are grouped as the comparisons of the paper:
the causal ViTs with PolyChirp's models, and the non-causal ViTs with PolyChirp's models (the same
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
    from sklearn.metrics import precision_recall_curve
    for group, models in groups("classification").items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
        for ax, (kind, key) in zip(axes, (("float", "float"), ("INT8", "quant"))):
            for m in models:
                y, z = [], []
                for d in run_dirs("classification", m):
                    s = np.load(d / ("scores.npz" if (d / "scores.npz").exists() else "results.npz"))
                    if f"y_score_{key}_test" in s:
                        y.append(s["y_true_test"].ravel())
                        z.append(s[f"y_score_{key}_test"].ravel())
                if y:
                    p, r, _ = precision_recall_curve(np.concatenate(y), np.concatenate(z))
                    ax.plot(r, p, label=f"{m} ({len(y)} runs)")
            ax.set(title=f"classification, test, {kind}", xlabel="recall", ylabel="precision", xlim=(0, 1), ylim=(0, 1.01))
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc="lower left")
        fig.tight_layout()
        fig.savefig(FIGURES / f"classification_pr_{group}.png", dpi=120)
        plt.close(fig)


def detection():
    grid = np.linspace(0, 1, 201)
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
                    tpr = np.mean([np.interp(grid, c[f"{split}/{kind}/roc_fpr"], c[f"{split}/{kind}/roc_tpr"]) for c in curves], axis=0)
                    boxes = np.mean([np.interp(grid, c[f"{split}/{kind}/time_recorded"], c[f"{split}/{kind}/boxes_captured"]) for c in curves], axis=0)
                    axes[0, col].plot(100 * grid, 100 * tpr, label=f"{m} ({len(curves)} runs)")
                    axes[1, col].plot(100 * grid, 100 * boxes, label=m)
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
    args = parser.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)
    if args.what == "labels":
        return labels(args.recording, args.start, args.end)
    if (RUNS / "classification").exists():
        classification()
    detection()


if __name__ == "__main__":
    main()
