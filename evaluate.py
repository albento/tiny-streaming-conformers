"""Turns every finished run of train.py into one CSV row per (run, test set, float / INT8), and computes
the embedded cost of every model.

    uv run evaluate.py          # -> results/classification_runs.csv, results/detection_runs.csv, results/costs.csv
    uv run evaluate.py --redo   # evaluate again the runs already in the CSV

Incremental: the rows already in the CSV are kept (runs evaluated on another machine, or whose folder
is gone); only the finished runs not in it yet are evaluated and added.

Classification (PolyChirp's evaluation, the same for every model): per-class thresholds maximizing F2
on the tuning set (train + val), then on the test clips: macro F2, macro average precision, and the
clip error (share of clips whose predicted set of species is not exactly the true one).

Detection, on the SSW test days, the first 8 s of every recording left out:
- segment ROC: common 1.47 s segments (buffers [24 + 23 k, 46 + 23 k]), positive when they overlap an
  annotation box. A model's segment score combines its decisions that judge the segment: streaming,
  the max of its 23 buffers' (each judges a part of the segment, which is positive if one part is);
  3 s windows, the mean of the two windows containing it (each judges the whole segment, plus 1.47 s
  before or after it: their max would let a bird in a neighbouring segment alone raise it). AUC and AP.
  Chance gives AUC 0.5 whatever the model's decision rate.
- triggered recording, as the AudioMoth's triggered firmware: a decision above the threshold opens a
  recording with a pre-roll (0.77 s, or the 3 s a window decision judged) and keeps it open TAIL_S
  after the last one. Over the thresholds (each run's own score quantiles): the share of the time
  recorded and the share of the annotation boxes a recording overlaps. Reported: the time needed to
  capture 80 / 90 % of the boxes, and the boxes captured with 20 / 30 % of the time.

Costs per model: parameters; multiply-accumulates per second of audio, front end included (window,
FFT, power, mel filterbank, PCEN; for the ViT the formula of vit.ViT.macs_per_buffer); flash and RAM
(the ViT: its C compiled for the Cortex-M4, with the stack and the max latency measured on the AudioMoth
once audiomoth.py has run; PolyChirp's models: their INT8 TFLite model's weights and
peak tensor arena, plus the 3 s of audio they need).
"""
import csv
import json
import math
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from paths import DATA, ROOT, RUNS

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # TensorFlow's C++ logs (PolyChirp sets the same, later)

RESULTS = ROOT / "results"
B, SR = 1024, 16000
BUFFER_S = B / SR
SKIP = math.ceil(8.0 / BUFFER_S)
TAIL_S, PREROLL_S = 1.5, 12 * BUFFER_S
SEGMENT, FIRST_SEGMENT = 23, 24


# --- classification -----------------------------------------------------------------------------------

def classification_rows(run: Path) -> list[dict]:
    from building.models import model_eval as M
    s = np.load(run / ("scores.npz" if (run / "scores.npz").exists() else "results.npz"))
    labels = [str(x) for x in s["label_names"]]
    rows = []
    for kind, key in (("float", "float"), ("int8", "quant")):
        if f"y_score_{key}_test" not in s:
            continue
        thresholds = M.pick_best_thresholds(s["y_true_train"], s[f"y_score_{key}_train"], labels, "best_f2")
        e = M.eval_with_thresholds(backend="keras" if kind == "float" else "tflite", label_names=labels,
                                   y_true=s["y_true_test"], y_score=s[f"y_score_{key}_test"], thresholds=thresholds)
        rows.append({"split": "test", "kind": kind, "f2": e.macro_f2, "clip_error": 1 - e.subset_accuracy,
                     "ap": average_precision_score(s["y_true_test"], s[f"y_score_{key}_test"], average="macro")})
    return rows


# --- detection ----------------------------------------------------------------------------------------

def boxes(collection: str) -> dict[str, np.ndarray]:
    """Per recording, the annotation boxes as buffer ranges [b0, b1), at least one buffer each."""
    out = defaultdict(list)
    with open(DATA / collection / "annotations.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            s, e = float(r["Start Time (s)"]), float(r["End Time (s)"])
            b0 = int(s / BUFFER_S)
            out[Path(r["Filename"]).stem].append((b0, max(math.ceil(e / BUFFER_S), b0 + 1)))
    return {k: np.array(v) for k, v in out.items()}


def span_sums(cumsum: np.ndarray, spans: np.ndarray) -> np.ndarray:
    return cumsum[spans[:, 1]] - cumsum[spans[:, 0]]


def segment_scores(z: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """(segment starts, score) from the decisions at buffers [s + min(window, 23) - 1, s + max(window, 23) - 1]
    (NaN where there is no decision): their max when each judges a part of the segment (window <= 23,
    streaming), their mean when each judges all of it (window > 23: the two 3 s windows containing it)."""
    lo, hi = min(window, SEGMENT) - 1, max(window, SEGMENT) - 1
    starts = np.arange(FIRST_SEGMENT, len(z) - hi, SEGMENT)
    starts = starts[starts >= SKIP]
    seg = z[starts[:, None] + np.arange(lo, hi + 1)]
    if window > SEGMENT:
        return starts, np.nanmean(seg, axis=1)
    return starts, np.where(np.isnan(seg), -np.inf, seg).max(axis=1)


def recorded(z: np.ndarray, threshold: float, preroll: int, tail: int) -> np.ndarray:
    """Buffers the triggered firmware records: a trigger at t records [t - preroll + 1, t + tail]."""
    trig = np.concatenate([[0], np.cumsum(z > threshold)])
    t = np.arange(len(z))
    rec = (trig[np.minimum(t + preroll, len(z))] - trig[np.maximum(t - tail, 0)]) > 0
    rec[:SKIP] = False
    return rec


def detection_rows(run: Path, splits: dict, box: dict) -> list[dict]:
    meta = json.loads((run / "train.json").read_text())
    window = meta.get("decision_span_buffers", 1)
    preroll, tail = max(round(PREROLL_S / BUFFER_S), window), round(TAIL_S / BUFFER_S)
    s = np.load(run / "scores.npz")
    rows, curves = [], {}
    for split, files in splits.items():
        for kind in ("float", "int8"):
            keys = [(c, f, f"{c}/{f}/{kind}") for c, f in files]
            if any(k not in s.files for _, _, k in keys):
                continue
            seg_y, seg_z = [], []
            z_all = [s[k] for _, _, k in keys]
            for (c, f, _), z in zip(keys, z_all):
                starts, score = segment_scores(z, window)
                mask = np.zeros(len(z), bool)
                for b0, b1 in box[c].get(f, []):
                    mask[b0:b1] = True
                seg_y.append(mask[starts[:, None] + np.arange(SEGMENT)].any(1))
                seg_z.append(score)
            seg_y, seg_z = np.concatenate(seg_y), np.concatenate(seg_z)
            thresholds = np.unique(np.nanquantile(np.concatenate(z_all), np.linspace(0.3, 0.9999, 200)))
            curve = []   # (share of the time recorded, share of the boxes overlapped) per threshold
            for th in thresholds:
                rec_t = tot_t = hit = n_box = 0
                for (c, f, _), z in zip(keys, z_all):
                    rec = recorded(z, th, preroll, tail)
                    b = np.minimum(box[c].get(f, np.zeros((0, 2), int)), len(z))   # a box may end in the last, partial buffer
                    b = b[(b[:, 1] > SKIP) & (b[:, 0] < len(z))]
                    rec_t, tot_t = rec_t + rec.sum(), tot_t + len(z) - SKIP
                    hit += int((span_sums(np.concatenate([[0], np.cumsum(rec)]), b) > 0).sum()) if len(b) else 0
                    n_box += len(b)
                curve.append((rec_t / tot_t, hit / n_box))
            share, recall = np.array(sorted(curve)).T
            recall = np.maximum.accumulate(recall)
            fpr, tpr, _ = roc_curve(seg_y, seg_z)
            curves |= {f"{split}/{kind}/roc_fpr": fpr, f"{split}/{kind}/roc_tpr": tpr,
                       f"{split}/{kind}/time_recorded": share, f"{split}/{kind}/boxes_captured": recall}
            rows.append({"split": split, "kind": kind, "segment_auc": roc_auc_score(seg_y, seg_z),
                         "segment_ap": average_precision_score(seg_y, seg_z),
                         "time_for_80": float(np.interp(0.8, recall, share)) if recall[-1] >= 0.8 else np.nan,
                         "time_for_90": float(np.interp(0.9, recall, share)) if recall[-1] >= 0.9 else np.nan,
                         "boxes_at_20": float(np.interp(0.2, share, recall)), "boxes_at_30": float(np.interp(0.3, share, recall))})
    np.savez_compressed(run / "curves.npz", **curves)   # for plot.py
    return rows


# --- costs --------------------------------------------------------------------------------------------

def stft_macs(n_fft: int, bins: int, filterbank_weights: int) -> float:
    """MACs of one STFT frame: window, real FFT (radix-2 complex FFT of n_fft / 2 points + split), power,
    filterbank, PCEN's running mean (3 per band)."""
    return n_fft + n_fft * math.log2(n_fft / 2) + 2 * n_fft + 2 * (n_fft // 2 + 1) + filterbank_weights + 3 * bins


def vit_costs(model: str, run: Path) -> dict:
    import train
    vit = train.make_vit(model, 1)
    front = vit.frames_per_buffer * stft_macs(256, 129, 0) / BUFFER_S
    window_tokens = 47 * vit.frames_per_buffer // vit.stem.time_stride
    out = {"parameters": sum(p.numel() for p in vit.parameters()),
           "macs_per_s_window": (vit.macs_per_buffer(window_tokens) * 47 + 47 * vit.frames_per_buffer * stft_macs(256, 129, 0)) / (23 * BUFFER_S)}
    if vit.causal:
        out["macs_per_s_stream"] = vit.macs_per_buffer(vit.context) / BUFFER_S + front
        export = next(iter(sorted(RUNS.glob(f"*/{model}/seed_*/export"))), None)
        if export is not None:
            import audiomoth   # measured on the device (audiomoth.py): RAM with the measured stack, max latency
            m = audiomoth.measured().get(model)
            out |= audiomoth.device_costs(model, export, m) if m else vit_memory(export)
    return out


def vit_memory(export: Path, stack_bytes: int = 1400) -> dict:
    """Flash (the model's C arrays) and static RAM (the ViT's and the front end's state) of the
    Cortex-M4 build, plus one buffer of audio and of spectrogram and the stack (default: 1.4 KB, an
    earlier measurement; audiomoth.py gives each model's own)."""
    import export as E
    sizes = {}
    with tempfile.TemporaryDirectory() as tmp:
        for src in ("vit_model.c", "pcen.c", "vit_inference.c", "frontend.c"):
            path = export / src if src in ("vit_model.c", "pcen.c") else E.FIRMWARE / src
            obj = Path(tmp) / (src + ".o")
            try:
                subprocess.run(["arm-none-eabi-gcc", "-c", "-O2", "-std=c99", "-mcpu=cortex-m4", "-mthumb", "-mfloat-abi=hard",
                                "-mfpu=fpv4-sp-d16", "-U__arm__", "-D__GNUC_PYTHON__", f"-I{E.FIRMWARE}", f"-I{export}",
                                f"-I{E.FIRMWARE}/CMSIS-NN/Include", f"-I{E.FIRMWARE}/CMSIS-DSP/Include",
                                f"-I{E.FIRMWARE}/CMSIS-DSP/PrivateInclude", "-o", str(obj), str(path)], check=True, capture_output=True)
            except (FileNotFoundError, subprocess.CalledProcessError):
                return {}  # no ARM toolchain
            text, data, bss = map(int, subprocess.run(["arm-none-eabi-size", str(obj)], capture_output=True, text=True).stdout.splitlines()[1].split()[:3])
            sizes[src] = (text + data, data + bss)
    return {"flash_kb": (sizes["vit_model.c"][0] + sizes["pcen.c"][0]) / 1024,
            "ram_kb": (sizes["vit_inference.c"][1] + sizes["frontend.c"][1] + 2 * B + 7 * 129 * 4 + stack_bytes) / 1024}


def baseline_costs(model: str, run: Path) -> dict:
    from keras import layers
    import baselines
    from building.models.bake import analyze_tflite
    from building.utils.mel_spec import build_rust_mel_matrix
    spec = baselines.specs()[model]
    keras_model = spec.builder(1)
    macs = 0
    for layer in keras_model.layers:   # the multiply-accumulates of every layer with weights
        o = layer.output.shape
        if isinstance(layer, layers.DepthwiseConv1D):
            macs += o[1] * layer.kernel_size[0] * o[2]
        elif isinstance(layer, layers.Conv2D):
            macs += o[1] * o[2] * layer.kernel_size[0] * layer.kernel_size[1] * layer.input.shape[-1] * layer.filters
        elif isinstance(layer, layers.Conv1D):
            macs += o[1] * layer.kernel_size[0] * layer.input.shape[-1] * layer.filters
        elif isinstance(layer, layers.Dense):
            macs += (int(np.prod(o[1:-1])) if len(o) > 2 else 1) * layer.input.shape[-1] * layer.units
        elif layer.__class__.__name__ == "TemporalAttentionPool":
            macs += 2 * layer.input.shape[1] * layer.input.shape[2]
    mel = spec.input_frontend
    _, fb = build_rust_mel_matrix(mel.n_bins, mel.step_window, mel.step_window, SR, mel.min_freq, mel.max_freq)
    window_macs = macs + mel.n_frames * stft_macs(mel.step_window, 0, int((fb > 0).sum()))
    out = {"parameters": keras_model.count_params(), "macs_per_s_window": window_macs / (23 * BUFFER_S),
           "macs_per_s_clip": window_macs / 3.0}
    tflite = next(iter(sorted(RUNS.glob(f"*/{model}/seed_*/model.tflite"))), None)
    if tflite is not None:
        stats = analyze_tflite(tflite)
        out |= {"flash_kb": stats.model_size_kb, "ram_kb": (stats.arena_size_kb or 0) + 2 * 48000 / 1024}
    return out


def wrennet_costs() -> dict:
    """WrenNet's parameters and MACs, counted as baseline_costs counts PolyChirp's models: the layers with
    weights (convolutions, GRU, linear), the attention pooling's weighted sum, the 32 kHz STFT and filter
    bank. No flash or RAM: no INT8 version."""
    import torch
    import baselines
    model = baselines.wrennet(1).eval()
    macs = 0

    def count(layer, args, out):
        nonlocal macs
        o = (out[0] if isinstance(out, tuple) else out).shape
        if isinstance(layer, torch.nn.Conv1d):
            macs += o[-1] * layer.kernel_size[0] * layer.in_channels // layer.groups * layer.out_channels
        elif isinstance(layer, torch.nn.Linear):
            macs += math.prod(o[:-1]) * layer.in_features * layer.out_features
        elif isinstance(layer, torch.nn.GRU):
            macs += o[1] * 3 * (layer.input_size + layer.hidden_size) * layer.hidden_size
    for layer in model.modules():
        layer.register_forward_hook(count)
    with torch.no_grad():
        model(torch.zeros(1, 2 * 48000))
    w = baselines.WRENNET
    frames = 2 * 48000 // w["hop_length"] + 1
    bank = model.combined_log_linear_spec._get_current_filter_bank("cpu")
    window_macs = macs + frames * w["hidden_dim"] + frames * stft_macs(w["n_fft"], 0, int((bank > 0).sum()))
    return {"parameters": sum(p.numel() for p in model.parameters()), "macs_per_s_window": window_macs / (23 * BUFFER_S),
            "macs_per_s_clip": window_macs / 3.0}


KEYS = ["experiment", "model", "seed"]


def merge(path: Path, new: list[dict], keys: list[str], replaced: set) -> None:
    """Writes path: its existing rows, minus those of the replaced keys, plus the new rows."""
    import pandas as pd
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=keys)
    old = old[[tuple(k) not in replaced for k in old[keys].itertuples(index=False)]]
    out = pd.concat([old, pd.DataFrame(new)], ignore_index=True) if new else old
    if len(out):
        out.sort_values(keys, kind="stable").to_csv(path, index=False)


def main():
    import argparse
    import pandas as pd
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--redo", action="store_true", help="evaluate again the runs already in the CSV (their rows replaced)")
    args = parser.parse_args()
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")   # nothing here needs a GPU
    RESULTS.mkdir(exist_ok=True)
    splits = json.loads((DATA / "splits.json").read_text()) if (DATA / "splits.json").exists() else None
    tests = {"ssw_test": [("ssw", f) for f in splits["ssw"]["test"]]} if splits else {}
    box = None
    for name in ("classification", "detection"):
        path = RESULTS / f"{name}_runs.csv"
        done = set(pd.read_csv(path)[KEYS].itertuples(index=False, name=None)) if path.exists() else set()
        rows, replaced = [], set()
        for run in sorted(RUNS.glob("*/*/seed_*")):
            experiment, model = run.parts[-3], run.parts[-2]
            if run.name.endswith(".partial") or not (run / "train.json").exists() or (experiment == "classification") != (name == "classification"):
                continue
            key = (experiment, model, int(run.name.split("_")[1]))
            if key in done and not args.redo and (name == "classification" or (run / "curves.npz").exists()):
                continue   # already in the CSV (detection: and its curves for plot.py)
            if name == "detection" and box is None:
                box = {"ssw": boxes("ssw")}
            found = classification_rows(run) if name == "classification" else detection_rows(run, tests, box)
            rows += [dict(zip(KEYS, key)) | r for r in found]
            replaced.add(key)
            print(f"{run}: {len(found)} rows", flush=True)
        merge(path, rows, KEYS, replaced)
        print(f"{path.name}: {len(replaced)} runs evaluated, the others kept")
    path = RESULTS / "costs.csv"
    known = set(pd.read_csv(path)["model"]) if path.exists() else set()
    costs = []
    for model in sorted({run.parts[-2] for run in RUNS.glob("*/*/seed_*")} - (set() if args.redo else known)):
        run = next(iter(sorted(RUNS.glob(f"*/{model}/seed_*"))))
        costs.append({"model": model, **(vit_costs(model, run) if re.match(r"vit_", model) else
                                         wrennet_costs() if model == "wrennet" else baseline_costs(model, run))})
    merge(path, costs, ["model"], {(c["model"],) for c in costs})

if __name__ == "__main__":
    main()
