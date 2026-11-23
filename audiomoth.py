"""Latency and RAM of the causal ViTs measured on the AudioMoth, from the CSV files of the latency
firmware (export.py), into results/audiomoth.csv and the columns of results/costs.csv.

    uv run export.py --all                   # one .bin per causal ViT, into runs/audiomoth/
    # flash each with the AudioMoth Flash App and let it record one period or more (16 kHz and full core
    # clock whatever the configuration); it writes one CSV file per period to the SD card, in place of the WAV
    uv run audiomoth.py <folder> [<folder> ...]   # the CSV files, searched recursively

Each file is assigned to a model by the model= tag of its header; every buffer of every file of a model
is pooled:
- latency of a 64 ms buffer: front end + ViT, in cycles / clock_hz, interrupts during it included (the
  wall-clock time, what real time is about). Max and mean; real time when the max is under 64 ms and no
  buffer was skipped (a buffer completed while both buffers were still waiting);
- RAM: evaluate.vit_memory's static RAM (the ViT's and the front end's .data + .bss, one buffer of audio
  and of spectrogram) with the deepest stack measured on the device (stack painting: the firmware's whole
  stack during the period, CSV writes included) in place of its 1.4 KB estimate.

results/audiomoth.csv keeps the measurements, so evaluate.py --redo writes the same costs again.
"""
import argparse
import re
from pathlib import Path

import pandas as pd

from paths import ROOT

RESULTS = ROOT / "results"
MEASURED = RESULTS / "audiomoth.csv"
BUFFER_MS = 1024 / 16000 * 1000
HEADER = re.compile(r"^# model=(?P<tag>.*?) depth=\d+ context=\d+ stem_time_stride=\d+ sample_rate=(?P<sr>\d+) clock_hz=(?P<clock>\d+)")


def read(path: Path) -> tuple[str, pd.DataFrame] | None:
    """(model, buffers with latency_ms) of one CSV file of the firmware; None if it is not one, or empty."""
    with open(path, errors="replace") as fh:
        m = HEADER.match(fh.readline())
    if m is None:
        return None
    if int(m["sr"]) != 16000:
        print(f"{path}: sample rate {m['sr']} Hz, no inference (the ViT needs 16 kHz): skipped")
        return None
    rows = pd.read_csv(path, comment="#")
    if rows.empty:
        print(f"{path}: no buffer: skipped")
        return None
    rows["latency_ms"] = (rows.frontend_cycles + rows.vit_cycles) / int(m["clock"]) * 1000
    return m["tag"].split()[0], rows   # the tag: "<model>" or "<model> seed <n>"


def measure(folders: list[Path]) -> pd.DataFrame:
    files = {}
    for folder in folders:
        for path in sorted(p for p in folder.rglob("*") if p.suffix.lower() == ".csv"):
            if (r := read(path)) is not None:
                files.setdefault(r[0], []).append(r[1])
    rows = []
    for model, parts in sorted(files.items()):
        b = pd.concat(parts, ignore_index=True)
        rows.append({"model": model, "files": len(parts), "buffers": len(b),
                     "latency_max_ms": b.latency_ms.max(), "latency_mean_ms": b.latency_ms.mean(),
                     "frontend_mean_ms": (b.latency_ms * b.frontend_cycles / (b.frontend_cycles + b.vit_cycles)).mean(),
                     "skipped_buffers": int(b.skipped_buffers.sum()), "stack_max_bytes": int(b.stack_max_bytes.max())})
    return pd.DataFrame(rows)


def measured() -> dict[str, dict]:
    """The measurements of results/audiomoth.csv, per model (empty if it does not exist)."""
    return pd.read_csv(MEASURED).set_index("model").to_dict("index") if MEASURED.exists() else {}


def device_costs(model: str, export: Path, m: dict) -> dict:
    """The costs.csv columns of one measured model."""
    import evaluate
    return evaluate.vit_memory(export, stack_bytes=m["stack_max_bytes"]) | {"latency_max_ms": m["latency_max_ms"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folders", type=Path, nargs="+", help="folders of the firmware's CSV files (searched recursively)")
    args = parser.parse_args()
    new = measure(args.folders)
    if new.empty:
        raise SystemExit("no CSV file of the latency firmware found")
    print(new.round(2).to_string(index=False))
    for r in new.itertuples():
        if r.skipped_buffers or r.latency_max_ms >= BUFFER_MS:
            print(f"{r.model}: NOT real time ({r.skipped_buffers} buffers skipped, max {r.latency_max_ms:.1f} ms)")
    RESULTS.mkdir(exist_ok=True)
    old = pd.read_csv(MEASURED) if MEASURED.exists() else pd.DataFrame(columns=["model"])
    out = pd.concat([old[~old.model.isin(new.model)], new], ignore_index=True).sort_values("model")
    out.to_csv(MEASURED, index=False)
    print(f"{MEASURED.name}: {len(new)} models measured, {len(out)} in all")

    costs_path = RESULTS / "costs.csv"
    if not costs_path.exists():
        raise SystemExit(f"{costs_path} missing: run evaluate.py first, then this again")
    from paths import RUNS
    costs = pd.read_csv(costs_path).set_index("model")
    for model, m in measured().items():
        export = next(iter(sorted(RUNS.glob(f"*/{model}/seed_*/export"))), None)
        if export is None or model not in costs.index:
            print(f"{model}: no export or not in costs.csv: left out of it")
            continue
        for k, v in device_costs(model, export, m).items():
            costs.loc[model, k] = v
    costs.reset_index().to_csv(costs_path, index=False)
    print(f"{costs_path.name}: ram_kb and latency_max_ms of the measured models")


if __name__ == "__main__":
    main()
