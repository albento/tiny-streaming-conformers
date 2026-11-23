"""Prepares the BirdSet soundscape collection of the detection task, from the Cornell Lab's
fully-annotated recordings on Zenodo (Kahl et al., CC BY 4.0):
  ssw  Sapsucker Woods, New York: 285 one-hour recordings, 50,760 boxes (train / val / test)

    uv run prepare_detection.py                 # download, convert, label, split; resumes where it stopped

Into data/<name>/ (git-ignored; or DATA_DIR of .env):
  audio/*.flac, annotations.csv      the Zenodo record (MD5-checked), the audio extracted
  audio16k/<file>.npy                int16 at 16 kHz (resampled from 32 kHz), memory-mappable
  labels.npz                         one label per 64 ms buffer: 1 bird, 0 no bird, 255 uncertain
  refined_annotations.csv            the labels as intervals in seconds (active / uncertain)
and data/splits.json: SSW split by recording day (all hours of a day on the same side; days shuffled
with seed 42, about 70 / 15 / 15 % of the files).

Labels: an annotation box often spans a whole bout (the annotators could merge calls of a species
less than 5 s apart). Inside every box, each 10 ms frame is kept as bird only where the box's
frequency band rises above the local noise floor; the rest of the box is "uncertain" and left out
of training and of the metrics. Outside every box: no bird. Method in README.md, "Labels".
"""
import csv
import hashlib
import json
import re
import shutil
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter
from scipy.signal import resample_poly

from paths import DATA

RECORDS = {"ssw": 7018484}  # Zenodo
SR, BUFFER, IGNORE = 16000, 1024, 255

# label refinement (10 ms frames)
N_FFT, HOP = 512, 160            # analysis STFT: 32 ms window, 10 ms hop
SMOOTH = (5, 3)                  # power averaged over 5 frames x 3 bins before the dB
BAND = (150.0, 8000.0)           # a box's band, clipped to what the 16 kHz audio holds
SNR_DB = 6.0                     # a frame is bird when its box's band is 6 dB above the noise floor
MIN_RUN, MAX_GAP, WIDEN = 5, 20, 5   # frames: runs < 50 ms dropped, gaps < 200 ms filled, 50 ms added each side
SHORT_BOX_S = 1.0                # boxes shorter than this are kept whole


# --- download, convert -----------------------------------------------------------------------------

def download(name: str) -> None:
    """Every file of the Zenodo record, MD5-checked, resuming partial downloads; the audio zip extracted."""
    folder = DATA / name
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "audio" / ".complete").exists():
        return
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{RECORDS[name]}", timeout=60) as r:
        files = json.load(r)["files"]
    for f in files:
        dest, part = folder / f["key"], folder / (f["key"] + ".part")
        done = part.stat().st_size if part.exists() else 0
        if not dest.exists():
            request = urllib.request.Request(f["links"]["self"], headers={"Range": f"bytes={done}-"} if done else {})
            with urllib.request.urlopen(request, timeout=120) as r, open(part, "ab" if done else "wb") as out:
                print(f"{name}: downloading {f['key']} ({f['size'] / 1e9:.1f} GB)", flush=True)
                shutil.copyfileobj(r, out, 1 << 20)
            part.rename(dest)
        md5 = hashlib.md5()
        with open(dest, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                md5.update(block)
        assert md5.hexdigest() == f["checksum"].removeprefix("md5:"), f"{dest}: MD5 mismatch, delete it and run again"
    with zipfile.ZipFile(folder / "soundscape_data.zip") as z:
        (folder / "audio").mkdir(exist_ok=True)
        for m in z.infolist():
            if not m.is_dir():
                with z.open(m) as src, open(folder / "audio" / Path(m.filename).name, "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
    (folder / "soundscape_data.zip").unlink()
    (folder / "audio" / ".complete").touch()


def to_16k(flac: Path) -> int:
    """FLAC (32 kHz) -> audio16k/<stem>.npy, int16; returns the sample count."""
    out = flac.parent.parent / "audio16k" / f"{flac.stem}.npy"
    if not out.exists():
        x, sr = sf.read(str(flac), dtype="float32")
        if sr != SR:
            x = resample_poly(x, 1, sr // SR)
        out.parent.mkdir(exist_ok=True)
        np.save(out, np.clip(np.round(x * 32768), -32768, 32767).astype(np.int16))
    return int(np.load(out, mmap_mode="r").shape[0])


# --- labels ----------------------------------------------------------------------------------------

def spectrogram_db(x: np.ndarray) -> np.ndarray:
    """[frames, bins] smoothed power in dB; frame i covers samples [i * HOP, i * HOP + N_FFT). Computed
    by minutes, overlapping by the smoothing's half-width. (Unsmoothed, the 3 largest bins of a frame
    of pure noise sit 8-10 dB above the floor: every silent gap would look like a bird.)"""
    n = (len(x) - N_FFT) // HOP + 1
    out = np.empty((n, N_FFT // 2 + 1), np.float32)
    window, pad = np.hanning(N_FFT + 1)[:-1], SMOOTH[0] // 2
    for i0 in range(0, n, 6000):
        i1 = min(n, i0 + 6000)
        j0, j1 = max(0, i0 - pad), min(n, i1 + pad)
        frames = np.lib.stride_tricks.sliding_window_view(x[j0 * HOP:(j1 - 1) * HOP + N_FFT].astype(np.float64), N_FFT)[::HOP]
        power = uniform_filter(np.abs(np.fft.rfft(frames * window, axis=1)) ** 2, SMOOTH, mode="nearest")
        out[i0:i1] = 10 * np.log10(np.maximum(power[i0 - j0:i1 - j0], 0) + 1e-3)
    return out


def noise_floor(db: np.ndarray, in_box: np.ndarray) -> np.ndarray:
    """[frames, bins]: per 10 s block, the median dB of the frames outside every box within +-30 s
    (the 20th percentile of all of them when that is under 10 s of frames)."""
    floor = np.empty_like(db)
    for b0 in range(0, len(db), 1000):
        near = slice(max(0, b0 - 2500), b0 + 3500)
        outside = db[near][~in_box[near]]
        floor[b0:b0 + 1000] = np.median(outside, 0) if len(outside) >= 1000 else np.percentile(db[near], 20, 0)
    return floor


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of the runs of True."""
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def bird_frames(snr: np.ndarray) -> np.ndarray:
    """Frames of one box above SNR_DB, cleaned: short runs dropped, short gaps filled, runs widened,
    the box's best frame always kept."""
    bird = snr >= SNR_DB
    for s, e in runs(bird):
        if e - s < MIN_RUN:
            bird[s:e] = False
    for s, e in runs(~bird):
        if 0 < s and e < len(bird) and e - s < MAX_GAP:
            bird[s:e] = True
    for s, e in runs(bird):
        bird[max(0, s - WIDEN):e + WIDEN] = True
    bird[np.argmax(snr)] = True
    return bird


def label_frames(audio: np.ndarray, boxes: list[tuple]) -> tuple[np.ndarray, np.ndarray]:
    """(bird, uncertain) masks per 10 ms frame of one recording, from its boxes (start, end, low, high)."""
    db = spectrogram_db(audio)
    n = len(db)
    frame = lambda t: min(n, int(round(t * SR / HOP)))  # noqa: E731
    spans = [(frame(s), max(frame(e), frame(s) + 1), lo, hi) for s, e, lo, hi in boxes]
    in_box = np.zeros(n, bool)
    for f0, f1, _, _ in spans:
        in_box[f0:f1] = True
    floor = noise_floor(db, in_box)
    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
    bird = np.zeros(n, bool)
    for (f0, f1, lo, hi), (s, e, _, _) in zip(spans, boxes):
        bins = np.flatnonzero((freqs >= max(lo, BAND[0])) & (freqs <= min(hi, BAND[1])))
        if bins.size == 0:              # inaudible at 16 kHz: uncertain
            continue
        if e - s < SHORT_BOX_S:
            bird[f0:f1] = True
            continue
        excess = db[f0:f1, bins] - floor[f0:f1, bins]
        snr = np.sort(excess, axis=1)[:, -3:].mean(axis=1)   # the box's 3 strongest bins
        bird[f0:f1] |= bird_frames(snr)
    return bird, in_box & ~bird


def label_recording(args) -> tuple[np.ndarray, list]:
    """Buffer labels of one recording and its intervals (start s, end s, kind)."""
    path, boxes = args
    audio = np.load(path, mmap_mode="r")
    bird, uncertain = label_frames(np.asarray(audio), boxes)
    buffers = len(audio) // BUFFER
    centre = (np.arange(len(bird)) * HOP + N_FFT // 2) // BUFFER   # the buffer holding each frame's centre
    labels = np.zeros(buffers, np.uint8)
    for mask, value in ((uncertain, IGNORE), (bird, 1)):           # bird wins over uncertain
        labels[np.unique(centre[mask & (centre < buffers)])] = value
    intervals = [(s * HOP / SR, e * HOP / SR, kind) for kind, mask in (("bird", bird), ("uncertain", uncertain))
                 for s, e in runs(mask)]
    return labels, intervals


def prepare(name: str) -> dict[str, int]:
    folder = DATA / name
    flacs = sorted((folder / "audio").glob("*.flac"))
    with ProcessPoolExecutor(8) as pool:
        sizes = dict(zip((f.stem for f in flacs), pool.map(to_16k, flacs)))
    if not (folder / "labels.npz").exists():
        boxes = defaultdict(list)
        with open(folder / "annotations.csv", newline="") as fh:
            for r in csv.DictReader(fh):
                boxes[Path(r["Filename"]).stem].append(tuple(float(r[k]) for k in ("Start Time (s)", "End Time (s)",
                                                                                     "Low Freq (Hz)", "High Freq (Hz)")))
        stems = sorted(sizes)
        with ProcessPoolExecutor(8) as pool:
            done = list(pool.map(label_recording, [(folder / "audio16k" / f"{s}.npy", boxes[s]) for s in stems]))
        np.savez_compressed(folder / "labels.npz", **{s: lab for s, (lab, _) in zip(stems, done)})
        with open(folder / "refined_annotations.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["Filename", "Start Time (s)", "End Time (s)", "Kind"])
            w.writerows((f"{s}.flac", f"{a:.2f}", f"{b:.2f}", k) for s, (_, iv) in zip(stems, done) for a, b, k in iv)
    return sizes


def split_by_day(stems: list[str]) -> dict[str, list[str]]:
    """SSW files <SITE>_<n>_<YYYYMMDD>_<time>: days shuffled (seed 42), taken in order up to ~70 / 15 / 15 %."""
    days = defaultdict(list)
    for s in stems:
        days[re.match(r"^[A-Za-z]+_\d+_(\d{8})_", s).group(1)].append(s)
    order = sorted(days)
    np.random.default_rng(42).shuffle(order)
    split, taken = {"train": [], "val": [], "test": []}, 0
    for day in order:
        name = "train" if taken < 0.70 * len(stems) else "val" if taken < 0.85 * len(stems) else "test"
        split[name] += sorted(days[day])
        taken += len(days[day])
    return split


def main():
    splits = {}
    for name in RECORDS:
        download(name)
        stems = sorted(prepare(name))
        splits[name] = split_by_day(stems)
        lab = np.concatenate(list(np.load(DATA / name / "labels.npz").values()))
        print(f"{name}: {len(stems)} recordings, buffers {100 * (lab == 1).mean():.1f} % bird, "
              f"{100 * (lab == IGNORE).mean():.1f} % uncertain, " + ", ".join(f"{k} {len(v)}" for k, v in splits[name].items()))
    (DATA / "splits.json").write_text(json.dumps(splits, indent=1))


if __name__ == "__main__":
    main()
