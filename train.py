"""Trains the models of the paper (models.yaml) and scores each run on its validation and test data.

    uv run train.py --list                                       # every (experiment, model) and its runs
    uv run train.py classification vit_s24_l3 --runs 10          # one model: seeds 0..9
    uv run train.py classification --runs 10                     # every model of one experiment
    uv run train.py --all                                        # everything, models.yaml's run count
    uv run train.py --all --parallel 4                           # 4 runs at a time on the GPU
    uv run train.py --clean                                      # delete the runs and results/

A run writes runs/<experiment>/<model>/seed_<n>/: train.log (its output), train.json (protocol, per-epoch
history: losses, validation AP, learning rate, time), scores.npz
(the scores evaluate.py reads), and for causal ViTs export/ (the INT8 C, firmware/). A finished run
is skipped, so an interrupted batch resumes where it stopped.

Experiments (models.yaml):
- classification: PolyChirp's Paris benchmark (10 species, 3 s clips), PolyChirp's data pipeline
  (clips, augmentation, class weights, threshold tuning set) for every model. The ViT reads its
  46 buffers (2.94 s) and decides as the device: an exponential moving average (0.5 s) of its buffer
  logits, max over the clip (a smooth max in training). Scores: every clip of the tuning set (train
  + val, clean) and of the test set, float and INT8 (the firmware C on the host, or PolyChirp's TFLite).
- detection_stream: bird / no bird for every 64 ms buffer of the soundscapes (prepare_detection.py),
  causal ViTs streamed over each one-hour recording as on the device (INT8) and by overlapping windows
  (float). Training: random 30 s regions (9 h of audio per epoch), the loss on every buffer after the
  model's receptive field, uncertain buffers left out.
- detection_window: bird / no bird in each 3 s window, a decision every 23 buffers (1.47 s), every
  model in the regime of the clip models: the same regions, decision times and window order for all;
  the ViT decides by a smooth max of its window's buffer logits.
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from paths import DATA, ROOT, RUNS

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # TensorFlow's C++ logs (PolyChirp sets the same, later)
# TensorFlow's GPU compiler (XLA) takes the first ptxas on the PATH: the venv's (CUDA 12.3, TensorFlow's own)
# before a system CUDA's, which may not support the GPU (CUDA 13 dropped the V100: "ptxas does not support CC 7.0")
_PTXAS = Path(np.__file__).resolve().parents[1] / "nvidia" / "cuda_nvcc" / "bin"
if _PTXAS.is_dir():
    os.environ["PATH"] = f"{_PTXAS}{os.pathsep}{os.environ.get('PATH', '')}"

CONFIG = yaml.safe_load((ROOT / "models.yaml").read_text())
P = CONFIG["protocol"]
SR, B = 16000, 1024
BUFFER_S = B / SR
IGNORE = 255
CLIP_SAMPLES = 46 * B                 # the ViT's part of a 3 s clip
WINDOW = 469                          # detection training regions: 30 s
EPOCH_HOURS, VAL_HOURS = 9.0, 4.5
EVERY, SPAN = 23, 47                  # detection_window: a decision every 23 buffers, on a 47-buffer (3 s) window
CLIP = 48000                          # PolyChirp's 3 s clip
SKIP_S = 8.0                          # metrics leave out the first 8 s of every recording


# --- models and runs ---------------------------------------------------------------------------------

def experiment_models(experiment: str) -> list[str]:
    e = CONFIG["experiments"][experiment]
    vit = [f"vit_{s}_l{n}{'' if c else '_nc'}" for c in e["vit"]["causal"] for s in e["vit"]["stems"] for n in CONFIG["vit"]["layers"]]
    return vit + e.get("baselines", [])


def run_dir(experiment: str, model: str, seed: int) -> Path:
    return RUNS / experiment / model / f"seed_{seed}"


def make_vit(model: str, n_classes: int):
    import vit
    stem, layers, nc = re.fullmatch(r"vit_(s\d+)_l(\d)(_nc)?", model).groups()
    return vit.ViT(n_classes, CONFIG["vit"]["stems"][stem], int(layers), causal=not nc, **CONFIG["vit"]["architecture"])


def seed_everything(seed: int, vit: bool) -> None:
    """Python, NumPy, TensorFlow; PyTorch for the ViT only: a PolyChirp run never imports PyTorch,
    whose CUDA 13 runtime, once in the process, breaks TensorFlow's CUDA 12 kernels ("invalid device function")."""
    import random
    import keras
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    if vit:
        import torch
        torch.manual_seed(seed)


RUN = ""  # "[experiment model s<seed>]", set by run_one: every line of a run starts with it (runs in parallel interleave)


def say(text: str) -> None:
    for line in text.split("\n"):
        print(f"{RUN} {line}" if RUN else line, flush=True)


def log_epoch(history: list, epoch: int, train_loss: float, val_loss: float, ap: float, best_ap: float, best_epoch: int,
              lr: float, seconds: float, improved: bool, new_lr: float, stop: bool) -> None:
    """One line per epoch, the same for every model (PolyChirp's too, baselines.py), and its history entry:
    the losses are the training loss (binary cross-entropy, sample weights included), the validation
    AP the selection score; * marks the epochs that improved it (the weights kept)."""
    history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_ap": ap, "lr": lr,
                    "seconds": seconds, "improved": improved})
    line = (f"e{epoch}/{P['epochs']}  train {train_loss:.4f}  val {val_loss:.4f}  AP {ap:.4f}{' *' if improved else '  '}  "
            f"best {best_ap:.4f} (e{best_epoch})  lr {lr:.1e}  {seconds:.0f}s")
    if new_lr < lr:
        line += f"  -> lr {new_lr:.1e}"
    if stop:
        line += f"\nstop at epoch {epoch}" + (f" (no gain for {P['patience']} epochs)" if epoch < P["epochs"] else "") + \
                f": best AP {best_ap:.4f} at epoch {best_epoch}"
    say(line)


class Protocol:
    """models.yaml's protocol on a score to minimize (minus the validation AP), the same for PyTorch
    and Keras: step() says whether the epoch improved, whether to halve the learning rate, whether to stop."""

    def __init__(self):
        self.best, self.best_epoch, self.bad, self.lr_bad, self.history = math.inf, 0, 0, 0, []
        self.t = time.time()

    def step(self, epoch: int, score: float, lr: float, train_loss: float, val_loss: float) -> tuple[bool, bool, bool]:
        improved = score < self.best - P["min_delta"]
        if improved:
            self.best, self.best_epoch, self.bad, self.lr_bad = score, epoch, 0, 0
        else:
            self.bad, self.lr_bad = self.bad + 1, self.lr_bad + 1
        reduce = self.lr_bad >= P["lr_patience"]
        if reduce:
            self.lr_bad = 0
        stop = self.bad >= P["patience"] or epoch + 1 == P["epochs"]
        seconds, self.t = time.time() - self.t, time.time()
        log_epoch(self.history, epoch + 1, train_loss, val_loss, -score, -self.best, self.best_epoch + 1, lr, seconds,
                  improved, max(lr * P["lr_factor"], P["lr_min"]) if reduce else lr, stop)
        return improved, reduce, stop


def bce(z: np.ndarray, y: np.ndarray) -> float:
    """Mean binary cross-entropy of logits z against targets y (the training loss, on the validation set)."""
    return float(np.mean(np.logaddexp(0, z) - y * z))


def note(t0: float, text: str) -> None:
    """A step of a run, with the time since its start."""
    m = int(time.time() - t0) // 60
    say(f"{m // 60}h{m % 60:02d}  {text}")


def torch_step(model, protocol: Protocol, opt, epoch: int, score: float, best: dict, **losses) -> bool:
    """Protocol step for a PyTorch model: keeps the best weights, halves the learning rate; True to stop."""
    improved, reduce, stop = protocol.step(epoch, score, opt.param_groups[0]["lr"], **losses)
    if improved:
        best.update({k: v.detach().clone() for k, v in model.state_dict().items()})
    if reduce:
        for g in opt.param_groups:
            g["lr"] = max(g["lr"] * P["lr_factor"], P["lr_min"])
    return stop


def save_json(path: Path, protocol: Protocol, t0: float, **extra) -> None:
    path.write_text(json.dumps({"protocol": P, "epochs": len(protocol.history), "best_epoch": protocol.best_epoch + 1,
                                "seconds": time.time() - t0, "history": protocol.history, **extra}, indent=1))


# --- classification ------------------------------------------------------------------------------------

def classification_data(seed: int):
    """PolyChirp's datasets of run `seed` (train_cell's): train (augmented), val, test, the clean train
    split (with val, the threshold tuning set), and the class names."""
    import baselines
    from building.polychirp_training.training import build_dataset_from_catalog, build_datasets, load_dataset_catalog
    catalog = load_dataset_catalog(baselines.COLLECTION)
    chosen, non_target = baselines.species(catalog, seed)
    rng = lambda: np.random.default_rng([seed, baselines.K, 0])  # noqa: E731
    train, val, test, labels, _ = build_datasets(catalog, chosen, non_target, P["batch_size"], rng(),
                                                 input_repr=None, training_augmentation=True)
    tune, _ = build_dataset_from_catalog(catalog, chosen, non_target, P["batch_size"], rng(), split="train",
                                         input_repr=None, augment=False)
    return train, val, test, tune, labels


def vit_input(audio) -> np.ndarray:
    """PolyChirp's audio batch ([-1, 1]) -> the ViT's int16-scale float input."""
    return np.asarray(audio, np.float32).reshape(len(audio), -1)[:, :CLIP_SAMPLES] * 32768


def train_vit_classification(model_name: str, seed: int, out: Path) -> None:
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import average_precision_score
    import export
    import vit
    t0 = time.time()
    dev = torch_device()
    train_ds, val_ds, test_ds, tune_ds, labels = classification_data(seed)
    model = make_vit(model_name, len(labels)).to(dev)
    r = CONFIG["vit"]["readout"]
    clip_logits = lambda z, train: vit.clip_logits(z, r["tau_s"], r["skip"], r["temperature"] if train else None)  # noqa: E731
    with torch.no_grad():  # PCEN's seed: the mean training power per bin (the firmware's model_pcen_initial_m)
        powers, n = [], 0
        for audio, *_ in tune_ds:
            powers.append(model.spectrogram.power(torch.from_numpy(vit_input(audio)).to(dev)).mean(dim=(0, 2)))
            n += len(audio)
            if n >= CONFIG["vit"]["pcen_seed_clips"]:
                break
        m0 = torch.stack(powers).mean(0)
    logits = lambda x: model(x, m_init=m0.expand(len(x), -1))  # noqa: E731
    opt = torch.optim.Adam(model.parameters(), lr=P["lr"], eps=P["adam_eps"])
    protocol, best = Protocol(), {}
    def loss_sum(x, y, w):   # PolyChirp's weighted loss, summed over the clips (the readout of training)
        return (w * F.binary_cross_entropy_with_logits(clip_logits(logits(x), True), y, reduction="none").mean(1)).sum()

    def tensors(audio, y, w):
        return (torch.from_numpy(a).to(dev) for a in (vit_input(audio), np.array(y, np.float32), np.array(w, np.float32)))
    note(t0, f"training on {torch_name(dev)}")
    for epoch in range(P["epochs"]):
        model.train()
        total, n = torch.zeros((), device=dev), 0
        for batch in train_ds:
            x, y, w = tensors(*batch)
            loss = loss_sum(x, y, w) / len(x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total, n = total + loss.detach() * len(x), n + len(x)
        model.eval()
        z, y, val_total, val_n = [], [], 0.0, 0
        with torch.no_grad():
            for batch in val_ds:
                x, t, w = tensors(*batch)
                z.append(clip_logits(logits(x), True).cpu().numpy())
                y.append(np.asarray(batch[1]))
                val_total, val_n = val_total + float(loss_sum(x, t, w)), val_n + len(x)
        if torch_step(model, protocol, opt, epoch, -average_precision_score(np.concatenate(y), np.concatenate(z), average="macro"), best,
                      train_loss=float(total) / n, val_loss=val_total / val_n):
            break
    model.load_state_dict(best)
    model.eval()
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "m_init": m0.cpu()}, out / "model.pt")
    binary = None
    if model.causal:  # INT8: calibration spectrograms from the first validation clips
        calib, n = [], 0
        for audio, *_ in val_ds:
            calib.append(torch.from_numpy(vit_input(audio)))
            n += len(audio)
            if n >= CONFIG["vit"]["calibration_clips"]:
                break
        with torch.no_grad():
            calib = torch.cat(calib)[:CONFIG["vit"]["calibration_clips"]].to(dev)
            spec = model.spectrogram(calib, m_init=m0.expand(len(calib), -1)).cpu()
        export.write_c(model, spec, m0, out / "export", tag=f"{model_name} seed {seed}")
        binary = export.build_host(out / "export")
        note(t0, "INT8 export done (export/)")

    def scores(datasets):
        y, fz, qz, pending = [], [], [], []

        def int8(pending):  # the firmware C on the host, buffer by buffer, then the same readout
            zb = export.host_logits(binary, np.clip(np.round(np.concatenate(pending)), -32768, 32767).astype(np.int16))
            qz.append(torch.sigmoid(clip_logits(torch.from_numpy(zb), False)).numpy())
        for ds in datasets:
            for audio, t, *_ in ds:   # the clean train split has no sample weights
                x = vit_input(audio)
                with torch.no_grad():
                    fz.append(torch.sigmoid(clip_logits(logits(torch.from_numpy(x).to(dev)), False)).cpu().numpy())
                y.append(np.asarray(t))
                if binary is not None:
                    pending.append(x)
                    if sum(map(len, pending)) >= 4096:
                        int8(pending)
                        pending = []
        if binary is not None and pending:
            int8(pending)
        return np.concatenate(y), np.concatenate(fz), np.concatenate(qz) if qz else None
    result = {"label_names": np.array(labels)}
    for split, datasets in (("train", (tune_ds, val_ds)), ("test", (test_ds,))):
        note(t0, f"scoring the {'tuning set (train + val)' if split == 'train' else 'test set'}, float" + (" and INT8" if binary else ""))
        y, f, q = scores(datasets)
        result |= {f"y_true_{split}": y, f"y_score_float_{split}": f} | ({f"y_score_quant_{split}": q} if q is not None else {})
    np.savez_compressed(out / "scores.npz", **result)
    save_json(out / "train.json", protocol, t0, parameters=sum(p.numel() for p in model.parameters()))


def train_classification(model: str, seed: int, out: Path) -> None:
    if model.startswith("vit_"):
        train_vit_classification(model, seed, out)
    else:
        import functools
        import baselines
        t0, history = time.time(), []
        baselines.train_classification(model, seed, out, P, functools.partial(log_epoch, history))  # results.npz: scores.npz's keys
        best = [h["epoch"] for h in history if h["improved"]]
        (out / "train.json").write_text(json.dumps({"protocol": P, "epochs": len(history), "best_epoch": best[-1] if best else None,
                                                    "seconds": time.time() - t0, "history": history, "trainer": "PolyChirp"}, indent=1))


# --- detection: data -------------------------------------------------------------------------------------

class Recordings:
    """The 16 kHz recordings (memory-mapped) and buffer labels of (collection, file) pairs."""

    def __init__(self, files: list[tuple[str, str]]):
        labels = {c: np.load(DATA / c / "labels.npz") for c in {c for c, _ in files}}
        self.names = [f"{c}/{f}" for c, f in files]
        self.audio = [np.load(DATA / c / "audio16k" / f"{f}.npy", mmap_mode="r") for c, f in files]
        self.labels = [labels[c][f] for c, f in files]
        self.buffers = np.array([len(y) for y in self.labels])

    def regions(self, n: int, rng) -> list[tuple[int, int]]:
        """(recording, first buffer) of n random 30 s regions, recordings weighted by duration."""
        rec = rng.choice(len(self.audio), size=n, p=self.buffers / self.buffers.sum())
        return [(i, int(rng.integers(0, self.buffers[i] - WINDOW + 1))) for i in rec]

    def clips(self, n: int, samples: int, rng) -> np.ndarray:
        """int16 [n, samples] at random places (calibration, PCEN seed)."""
        rec = rng.choice(len(self.audio), size=n, p=self.buffers / self.buffers.sum())
        return np.stack([self.audio[i][(s := int(rng.integers(0, self.buffers[i] * B - samples))):s + samples] for i in rec])

    def region_batch(self, regions) -> tuple[np.ndarray, np.ndarray]:
        """int16 audio [n, WINDOW * B] and labels [n, WINDOW]."""
        return (np.stack([self.audio[i][b * B:(b + WINDOW) * B] for i, b in regions]),
                np.stack([self.labels[i][b:b + WINDOW] for i, b in regions]))


def detection_data() -> dict[str, Recordings]:
    s = json.loads((DATA / "splits.json").read_text())
    return {"train": Recordings([("ssw", f) for f in s["ssw"]["train"]]), "val": Recordings([("ssw", f) for f in s["ssw"]["val"]]),
            "ssw_test": Recordings([("ssw", f) for f in s["ssw"]["test"]]), "sne_test": Recordings([("sne", f) for f in s["sne"]["test"]])}


def n_regions(hours: float) -> int:
    return round(hours * 3600 / (WINDOW * BUFFER_S))


def window_targets(labels: np.ndarray, rng=None):
    """(region, decision buffer, label) of the decisions in regions [R, WINDOW]: a decision at buffer b
    judges (b - SPAN, b], bird if one of its buffers is, else left out if one is uncertain. With rng:
    a decision every EVERY buffers from a random phase per region; without: at every buffer."""
    first = SPAN - 1
    if rng is None:
        r, b = (a.ravel() for a in np.meshgrid(np.arange(len(labels)), np.arange(first, WINDOW), indexing="ij"))
    else:
        b = [np.arange(first + p, WINDOW, EVERY) for p in rng.integers(0, EVERY, len(labels))]
        r = np.concatenate([np.full(len(x), i) for i, x in enumerate(b)])
        b = np.concatenate(b)
    seg = labels[r[:, None], b[:, None] - np.arange(SPAN)]
    y = np.where((seg == 1).any(1), 1, np.where((seg == IGNORE).any(1), IGNORE, 0))
    keep = y != IGNORE
    return r[keep], b[keep], y[keep].astype(np.float32)


def window_audio(audio: np.ndarray, b: np.ndarray, r: np.ndarray | None = None, samples: int = SPAN * B) -> np.ndarray:
    """The `samples` samples ending with buffers b (of regions r of audio [R, n], or of one recording)."""
    idx = ((b + 1) * B - samples)[:, None] + np.arange(samples)
    return audio[r[:, None], idx] if r is not None else np.asarray(audio)[idx]


# --- detection: models -------------------------------------------------------------------------------------

class Detector:
    """One model of a detection experiment, behind the same calls for the ViT (PyTorch) and the
    baselines (Keras): fit_batch, logits (float), int8 (after quantize), state for the best weights."""

    def __init__(self, name: str, experiment: str, train: Recordings, seed: int):
        self.name, self.window = name, experiment == "detection_window"
        self.vit = name.startswith("vit_")
        if self.vit:
            import torch
            self.torch, self.dev = torch, torch_device()
            self.model = make_vit(name, 1).to(self.dev)
            clips = train.clips(CONFIG["vit"]["pcen_seed_clips"], CLIP_SAMPLES, np.random.default_rng(0))
            with torch.no_grad():  # PCEN's seed: the mean training power per bin
                self.m0 = torch.stack([self.model.spectrogram.power(torch.from_numpy(c.astype(np.float32)).to(self.dev)).mean(dim=(0, 2))
                                       for c in np.array_split(clips, 8)]).mean(0)
            self.opt = torch.optim.Adam(self.model.parameters(), lr=P["lr"], eps=P["adam_eps"])
            self.samples = SPAN * B
        else:
            import keras
            import baselines
            self.model, self.features = baselines.build_detector(name)
            self.model.compile(optimizer=keras.optimizers.Adam(P["lr"], epsilon=P["adam_eps"]), loss="binary_crossentropy")
            self.samples = CLIP
        self.binary = self.tflite = None

    # the ViT's outputs: buffer logits [n, buffers]; a window decision: their smooth max
    def _vit_buffers(self, x):
        return self.model(x, m_init=self.m0.expand(len(x), -1))[..., 0]

    def _vit_window(self, z):
        import vit
        return vit.window_logit(z[..., None], CONFIG["vit"]["readout"]["temperature"])

    def fit_batch(self, audio: np.ndarray, y: np.ndarray, warmup: int = 0) -> float:
        """One step on int16 audio [n, samples] with targets: per buffer [n, buffers] (stream) or per window [n]."""
        if self.vit:
            torch, F = self.torch, self.torch.nn.functional
            self.model.train()
            x, y = torch.from_numpy(audio.astype(np.float32)).to(self.dev), torch.from_numpy(y.astype(np.float32)).to(self.dev)
            z = self._vit_buffers(x)
            if self.window:
                loss = F.binary_cross_entropy_with_logits(self._vit_window(z), y)
            else:
                z, y = z[:, warmup:], y[:, warmup:]
                keep = (y != IGNORE).float()
                loss = F.binary_cross_entropy_with_logits(z, y.clamp(max=1), weight=keep, reduction="sum") / keep.sum().clamp(min=1)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
            return loss.item()
        return float(np.ravel(self.model.train_on_batch(self.features((audio / 32768).astype(np.float32)), y[:, None]))[0])

    def logits(self, audio: np.ndarray) -> np.ndarray:
        """Float logits: per buffer [n, buffers] (stream) or per window [n]."""
        if self.vit:
            self.model.eval()
            with self.torch.no_grad():
                z = self._vit_buffers(self.torch.from_numpy(audio.astype(np.float32)).to(self.dev))
                return (self._vit_window(z) if self.window else z).cpu().numpy()
        p = np.asarray(self.model.predict_on_batch(self.features((audio / 32768).astype(np.float32))))[:, 0]
        return np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))

    def state(self):
        return {k: v.detach().clone() for k, v in self.model.state_dict().items()} if self.vit else self.model.get_weights()

    def load(self, state) -> None:
        self.model.load_state_dict(state) if self.vit else self.model.set_weights(state)

    def step(self, protocol: Protocol, epoch: int, score: float, best: list, **losses) -> bool:
        lr = self.opt.param_groups[0]["lr"] if self.vit else float(self.model.optimizer.learning_rate.numpy())
        improved, reduce, stop = protocol.step(epoch, score, lr, **losses)
        if improved:
            best[:] = [self.state()]
        if reduce:
            if self.vit:
                for g in self.opt.param_groups:
                    g["lr"] = max(g["lr"] * P["lr_factor"], P["lr_min"])
            else:
                self.model.optimizer.learning_rate.assign(max(lr * P["lr_factor"], P["lr_min"]))
        return stop

    def quantize(self, calib_audio: np.ndarray, out: Path) -> None:
        """INT8 version of a causal ViT (export.py) or of a baseline (PolyChirp's TFLite recipe)."""
        if self.vit:
            if not self.model.causal:
                return
            import export
            with self.torch.no_grad():
                x = self.torch.from_numpy(calib_audio[:CONFIG["vit"]["calibration_clips"]].astype(np.float32)).to(self.dev)
                spec = self.model.spectrogram(x, m_init=self.m0.expand(len(x), -1)).cpu()
            export.write_c(self.model, spec, self.m0, out / "export", tag=f"{self.name}")
            self.binary = export.build_host(out / "export")
        else:
            import tensorflow as tf
            from building.models.bake import bake_model
            x = np.asarray(self.features((calib_audio / 32768).astype(np.float32)))
            bake_model(self.model, tf.data.Dataset.from_tensor_slices((x, np.zeros((len(x), 1), np.float32))).batch(32),
                       out / "model.tflite", n_representative=len(x), verbose=False)
            self.tflite = out / "model.tflite"

    def int8_logits(self, audio: np.ndarray, whole: bool = False) -> np.ndarray | None:
        """INT8 logits, like logits(); whole: one recording streamed from its start (stream ViT)."""
        if self.binary is not None:
            import export
            z = export.host_logits(self.binary, audio[None] if whole else audio, processes=1 if whole else 8)[..., 0]
            return z[0] if whole else (self._vit_window(self.torch.from_numpy(z)).numpy() if self.window else z)
        if self.tflite is not None:
            return tflite_logits(self.tflite, np.asarray(self.features((audio / 32768).astype(np.float32))))
        return None


def available_cpus() -> int:
    """The CPUs this process may use: its affinity (SLURM's cpuset), capped by a cgroup CPU quota
    (containers), shared between the runs in parallel (os.cpu_count() counts the whole node)."""
    n = len(os.sched_getaffinity(0))
    for quota, period in (("/sys/fs/cgroup/cpu.max", None), ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us")):
        try:
            q, t = Path(quota).read_text().split()[:2] if period is None else (Path(quota).read_text(), Path(period).read_text())
            if q.strip() not in ("max", "-1"):
                n = min(n, max(1, int(q) // int(t)))
        except (OSError, ValueError):
            pass
    return max(1, n // int(os.environ.get("TSC_PARALLEL", "1")))


def tflite_logits(path: Path, x: np.ndarray, threads: int | None = None) -> np.ndarray:
    """Logits of an INT8 TFLite model (batch of 1 baked in): one single-threaded interpreter per thread."""
    import tensorflow as tf
    threads = threads or min(16, available_cpus())

    def run(chunk):
        it = tf.lite.Interpreter(model_path=str(path), num_threads=1)
        it.allocate_tensors()
        i, o = it.get_input_details()[0], it.get_output_details()[0]
        (si, zi), (so, zo) = i["quantization"], o["quantization"]
        q = np.clip(np.round(chunk / si) + zi, -128, 127).astype(np.int8)
        p = []
        for k in range(len(q)):
            it.set_tensor(i["index"], q[k:k + 1])
            it.invoke()
            p.append((float(it.get_tensor(o["index"])[0, 0]) - zo) * so)
        return np.array(p)
    with ThreadPoolExecutor(threads) as pool:
        p = np.concatenate(list(pool.map(run, np.array_split(x, threads))))
    return np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))


# --- detection: training and scoring ---------------------------------------------------------------------

def train_detection(experiment: str, model_name: str, seed: int, out: Path) -> None:
    from sklearn.metrics import average_precision_score
    t0 = time.time()
    data = detection_data()
    det = Detector(model_name, experiment, data["train"], seed)
    warmup = 0 if det.window else det.model.receptive_field()
    regions_rng, order_rng = np.random.default_rng(seed), np.random.default_rng([seed, 1])
    val_audio, val_labels = data["val"].region_batch(data["val"].regions(n_regions(VAL_HOURS), np.random.default_rng(1)))
    if det.window:
        vr, vb, vy = window_targets(val_labels)
    protocol, best = Protocol(), []
    note(t0, f"training on {torch_name(det.dev) if det.vit else tensorflow_name()}: {EPOCH_HOURS} h of audio per epoch, "
             f"{len(data['train'].names)} train / {len(data['val'].names)} val recordings")
    for epoch in range(P["epochs"]):
        audio, labels = data["train"].region_batch(data["train"].regions(n_regions(EPOCH_HOURS), regions_rng))
        losses = []
        if det.window:  # one window per decision, shuffled across regions
            r, b, y = window_targets(labels, order_rng)
            order = order_rng.permutation(len(r))
            for k in range(0, len(order), P["batch_size"]):
                i = order[k:k + P["batch_size"]]
                losses.append(det.fit_batch(window_audio(audio, b[i], r[i], det.samples), y[i]))
            z = np.concatenate([det.logits(window_audio(val_audio, vb[k:k + 512], vr[k:k + 512], det.samples)) for k in range(0, len(vr), 512)])
            y_val = vy
        else:           # whole regions, the loss on every buffer after the receptive field
            for k in range(0, len(audio), P["batch_size"]):
                losses.append(det.fit_batch(audio[k:k + P["batch_size"]], labels[k:k + P["batch_size"]], warmup))
            z = np.concatenate([det.logits(val_audio[k:k + 32])[:, warmup:] for k in range(0, len(val_audio), 32)]).ravel()
            y_val = val_labels[:, warmup:].ravel()
            z, y_val = z[y_val != IGNORE], y_val[y_val != IGNORE]
        if det.step(protocol, epoch, -average_precision_score(y_val, z), best, train_loss=float(np.mean(losses)), val_loss=bce(z, y_val)):
            break
    det.load(best[0])
    out.mkdir(parents=True, exist_ok=True)
    note(t0, "INT8 version" + (" (export/)" if det.vit else " (model.tflite)"))
    # INT8 calibration from the validation days: 240 windows for the ViT (export.py), 2000 for PolyChirp's TFLite recipe
    det.quantize(data["val"].clips(CONFIG["vit"]["calibration_clips"] if det.vit else 2000, det.samples, np.random.default_rng(2)), out)
    scores = {}
    for split in ("val", "ssw_test", "sne_test"):
        d = data[split]
        note(t0, f"scoring {split}: {len(d.names)} recordings, float" + (" and INT8" if det.binary or det.tflite else ""))
        for name, audio, n in zip(d.names, d.audio, d.buffers):
            if det.window:  # decisions at buffers 46, 69, ...; NaN between them
                b = np.arange(SPAN - 1, n, EVERY)
                f, q = np.full(n, np.nan, np.float32), np.full(n, np.nan, np.float32)
                for k in range(0, len(b), 4096):
                    x = window_audio(audio, b[k:k + 4096], samples=det.samples)
                    f[b[k:k + 4096]] = np.concatenate([det.logits(x[j:j + 512]) for j in range(0, len(x), 512)])
                    zq = det.int8_logits(x)
                    if zq is not None:
                        q[b[k:k + 4096]] = zq
            else:           # float: overlapping 30 s windows, each buffer with its receptive field of past
                f = np.zeros(n, np.float32)
                starts = list(range(0, n - WINDOW + 1, WINDOW - warmup)) + [n - WINDOW]
                for s in starts:
                    z = det.logits(np.asarray(audio[s * B:(s + WINDOW) * B])[None])[0]
                    first = 0 if s == 0 else warmup
                    f[s + first:s + WINDOW] = z[first:]
                q = det.int8_logits(np.asarray(audio[:n * B]), whole=True)   # INT8: the whole hour, as the device
            scores[f"{name}/float"] = f
            if q is not None and not np.isnan(q).all():
                scores[f"{name}/int8"] = q
    np.savez_compressed(out / "scores.npz", **scores)
    save_json(out / "train.json", protocol, t0, warmup_buffers=warmup, decision_every_buffers=EVERY if det.window else 1,
              decision_span_buffers=SPAN if det.window else 1)


# --- command line -------------------------------------------------------------------------------------

def torch_device():
    """The GPU if this PyTorch build has kernels for it, else the CPU. PyTorch's CUDA 13 builds start at
    compute capability 7.5: a V100 (7.0) is seen but fails at the first kernel."""
    import torch
    if not torch.cuda.is_available():
        return torch.device("cpu")
    major, minor = torch.cuda.get_device_capability(0)
    for arch in torch.cuda.get_arch_list():   # sm_XY runs on X.Y and later minors; compute_XY (PTX) on X.Y and later
        kind, cc = arch.split("_")
        if (int(cc) // 10 == major and int(cc) % 10 <= minor) or (kind == "compute" and divmod(int(cc), 10) <= (major, minor)):
            return torch.device("cuda", 0)
    say(f"{torch.cuda.get_device_name(0)} (compute capability {major}.{minor}): this PyTorch build has no kernels "
        f"for it ({', '.join(torch.cuda.get_arch_list())}), the ViT trains on the CPU")
    return torch.device("cpu")


def torch_name(dev) -> str:
    import torch
    return f"the GPU ({torch.cuda.get_device_name(dev)}, PyTorch)" if dev.type == "cuda" else "the CPU (PyTorch)"


def tensorflow_name() -> str:
    import tensorflow as tf
    return f"the {'GPU' if tf.config.list_logical_devices('GPU') else 'CPU'} (TensorFlow)"


class Tee:
    """stdout to the terminal and to the run's train.log."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for st in self.streams:
            st.write(text)
        return len(text)

    def flush(self):
        for st in self.streams:
            st.flush()


def tensorflow_gpu_ok() -> bool:
    """False on GPUs newer than PolyChirp's TensorFlow 2.16 (compiled up to compute capability 9.0):
    Blackwell (12.x) fails at the first kernel. Asked to TensorFlow, before it uses the GPU (not to
    PyTorch: see seed_everything)."""
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return True
    details = tf.config.experimental.get_device_details(gpus[0])
    if details.get("compute_capability", (0, 0))[0] >= 10:
        say(f"{details.get('device_name', 'GPU')}: TensorFlow 2.16 cannot run on this GPU, PolyChirp's models train on the CPU")
        return False
    return True


def run_one(experiment: str, model: str, seed: int) -> None:
    out = run_dir(experiment, model, seed)
    if (out / "train.json").exists():
        print(f"{out}: done")
        return
    global RUN
    RUN = f"[{experiment} {model} s{seed}]"
    tmp = out.with_name(out.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    sys.stdout = Tee(sys.__stdout__, open(tmp / "train.log", "w"))
    say(f"=== {experiment} {model} seed {seed}")
    if model.startswith("vit_") or not tensorflow_gpu_ok():
        # the ViT: TensorFlow only reads PolyChirp's data, the GPU is PyTorch's. PolyChirp's models on a GPU
        # TensorFlow 2.16 has no kernels for (Blackwell, compute capability 12): on the CPU, slower.
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
    seed_everything(seed, model.startswith("vit_"))
    t0 = time.time()
    (train_classification(model, seed, tmp) if experiment == "classification" else train_detection(experiment, model, seed, tmp))
    tmp.rename(out)
    note(t0, f"=== done: {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", nargs="?", choices=list(CONFIG["experiments"]))
    parser.add_argument("models", nargs="*", help="model ids (default: every model of the experiment)")
    parser.add_argument("--runs", type=int, default=CONFIG["runs"], help="seeds 0 .. runs - 1")
    parser.add_argument("--all", action="store_true", help="every experiment and model")
    parser.add_argument("--parallel", type=int, default=1, help="runs at a time (separate processes)")
    parser.add_argument("--list", action="store_true", help="list the runs and whether they are done")
    parser.add_argument("--clean", action="store_true", help="delete the runs and results/")
    parser.add_argument("--one", nargs=3, metavar=("EXPERIMENT", "MODEL", "SEED"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.clean:
        for d in [RUNS / e for e in CONFIG["experiments"]] + [ROOT / "results"]:
            shutil.rmtree(d, ignore_errors=True)
        return
    if args.one:
        return run_one(args.one[0], args.one[1], int(args.one[2]))
    experiments = list(CONFIG["experiments"]) if args.all or not args.experiment else [args.experiment]
    jobs = [(e, m, s) for e in experiments for m in (args.models if args.experiment and args.models else experiment_models(e))
            for s in range(args.runs)]
    if args.list or not (args.all or args.experiment):
        for e, m, s in jobs:
            print(f"{e:18s} {m:18s} seed {s}  {'done' if (run_dir(e, m, s) / 'train.json').exists() else '-'}")
        return
    jobs = [j for j in jobs if not (run_dir(*j) / "train.json").exists()]
    # every run in its own process (GPU memory, TensorFlow state); faulthandler: a crash in C prints its Python stack
    command = lambda j: [sys.executable, "-X", "faulthandler", __file__, "--one", j[0], j[1], str(j[2])]  # noqa: E731
    os.environ["TSC_PARALLEL"] = str(args.parallel)   # the runs share the job's CPUs (available_cpus)
    with ThreadPoolExecutor(args.parallel) as pool:
        codes = list(pool.map(lambda j: subprocess.run(command(j)).returncode, jobs))
    failed = [(j, c) for j, c in zip(jobs, codes) if c]
    print(f"{len(jobs) - len(failed)} runs done" + (f", {len(failed)} failed:" if failed else ""))
    for (e, m, seed), c in failed:
        print(f"  {e} {m} seed {seed}: " + (f"exit code {c} (its traceback is above)" if c > 0 else exit_signal(-c)))


def exit_signal(number: int) -> str:
    """Why a run was killed: no Python traceback is printed in that case."""
    import signal
    name = signal.Signals(number).name if number in signal.Signals._value2member_map_ else f"signal {number}"
    hint = {"SIGKILL": "killed from outside: usually out of memory (a SLURM job's memory limit, or the kernel's OOM killer)",
            "SIGSEGV": "crash in C code (see the faulthandler stack above)", "SIGABRT": "abort in C code (see above)",
            "SIGTERM": "terminated (time limit of the job?)"}.get(name, "")
    return f"killed by {name}" + (f": {hint}" if hint else "")


if __name__ == "__main__":
    main()
