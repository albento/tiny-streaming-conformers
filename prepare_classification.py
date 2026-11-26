"""Builds PolyChirp's Paris dataset (the classification benchmark) with PolyChirp's own code, into the
submodule's git-ignored folders (polychirp/raw_dataset/, polychirp/datasets/scale_49_2_r50/; or links to
CLASSIFICATION_DATA_DIR of .env), as its
notebook building/geographic_scale/11_scale_dataset_build.ipynb does, with three changes:
- the 10 target species are the paper's, fixed below (the notebook picks the area's most recorded
  species, which changes as recordings are added), and so are the other species of the area, whose
  clips are pooled into the negative class "non_target";
- Xeno-Canto only: the eBird / Macaulay Library media are behind a bot check;
- AudioSet's ambient clips (non_target, and the background noise of the augmentation) come from the
  Hugging Face copy of AudioSet (agkphysics/AudioSet), not from YouTube, which blocks downloads after
  a few hundred clips. Same classes, quotas and 3 s windows as PolyChirp (building/data/audioset.py).

    uv run prepare_classification.py            # every step; resumes where it stopped

Needs XC_API_KEY (Xeno-Canto API key) in polychirp/.env. Xeno-Canto's content changes over time:
a new build has the same species and budgets, not the same clips.
"""
import asyncio
import heapq
import io
import multiprocessing as mp
import random
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from pathlib import Path

import paths

COLLECTION = "scale_49_2_r50"  # PolyChirp's slug: 50 km around Paris (48.8566, 2.3522)
TARGET_SPECIES = ["Psittacula krameri", "Hippolais polyglotta", "Parus major", "Dendrocoptes medius",
                  "Certhia brachydactyla", "Fringilla coelebs", "Curruca communis", "Troglodytes troglodytes",
                  "Sylvia atricapilla", "Regulus ignicapilla"]
OTHER_SPECIES = ["Anthus spinoletta", "Erithacus rubecula", "Acrocephalus palustris", "Poecile palustris",
                 "Phylloscopus collybita", "Sitta europaea", "Tyto alba", "Locustella luscinioides",
                 "Corvus corone corone", "Locustella naevia", "Turdus merula", "Pica pica", "Picus viridis",
                 "Corvus corone", "Turdus philomelos", "Luscinia megarhynchos", "Phoenicurus phoenicurus",
                 "Lophophanes cristatus", "Dendrocopos major", "Cyanistes caeruleus", "Sylvia borin",
                 "Emberiza schoeniclus", "Anthus spinoletta spinoletta"]
# the notebook's budgets
TARGET_CLIPS, OTHER_CLIPS, OTHER_CAP = 5000, 250, 300
AUDIOSET_PER_CLASS, AUDIOSET_TOTAL, AUDIOSET_MAX_GB = 250, 15000, 300
HF_REPO = "datasets/agkphysics/AudioSet"
HF_SPLITS = {"unbal_train": "unbalanced_train", "bal_train": "balanced_train", "eval": "eval"}  # HF -> AudioSet CSV


def write_manifest():
    from building.polychirp_training.dataset import Audio, Budgets, DatasetInfo, Place, write_dataset_info
    write_dataset_info(DatasetInfo(
        collection=COLLECTION, place=Place(name="Paris", lat=48.8566, lon=2.3522, radius_km=50), n_targets=10,
        target_species=TARGET_SPECIES, other_species=OTHER_SPECIES, audio=Audio(),
        budgets=Budgets(target_clips_per_species=TARGET_CLIPS, non_target_clips_per_species=OTHER_CLIPS,
                        non_target_per_species_cap=OTHER_CAP, audioset_clips_per_class=AUDIOSET_PER_CLASS,
                        audioset_max_total=AUDIOSET_TOTAL)))


async def download_birds():
    """Xeno-Canto clips of every species, then more clips of the other species until BirdNET has
    mined enough "no bird" clips from them (the notebook's "boost no_bird" cell)."""
    from building.data.download import BIRDNET_NO_BIRD_DIR, NO_BIRD_CAP, download_and_process
    from building.data.sources import XCSource
    with XCSource() as xc:
        for species, quota in [(s, TARGET_CLIPS) for s in TARGET_SPECIES] + [(s, OTHER_CLIPS) for s in OTHER_SPECIES]:
            await download_and_process(species, quota, [xc])
        no_bird = lambda: len(list(BIRDNET_NO_BIRD_DIR.glob("*.wav"))) if BIRDNET_NO_BIRD_DIR.exists() else 0  # noqa: E731
        quota = OTHER_CLIPS
        while no_bird() < NO_BIRD_CAP:
            quota *= 2
            before = no_bird()
            for species in OTHER_SPECIES:
                await download_and_process(species, quota, [xc], verbose=False)
            if no_bird() == before:  # Xeno-Canto has no more recordings of these species
                break


# --- AudioSet from Hugging Face ---------------------------------------------------------------------
# The HF copy is parquet in row groups of 100 clips; one clip costs its whole row group. Row groups are
# picked greedily, the one holding the most still-needed clips first.

def hf_fs():
    from huggingface_hub import HfFileSystem
    return HfFileSystem()


def row_groups(split: str) -> list[dict]:
    """Every row group of one HF split: file, index, size, and its clips' video ids."""
    import pyarrow.parquet as pq

    def index(path):
        with hf_fs().open(path, block_size=1 << 16) as f:
            pf = pq.ParquetFile(f)
            ids = pf.read(columns=["video_id"]).column("video_id").to_pylist()
            out, start = [], 0
            for i in range(pf.metadata.num_row_groups):
                rg = pf.metadata.row_group(i)
                size = sum(rg.column(j).total_compressed_size for j in range(rg.num_columns))
                out.append({"path": path, "rg": i, "bytes": size, "ids": ids[start:start + rg.num_rows]})
                start += rg.num_rows
            return out
    files = [f for f in hf_fs().ls(f"{HF_REPO}/data/{split}", detail=False) if f.endswith(".parquet")]
    with ThreadPoolExecutor(8) as ex:
        return [g for gs in ex.map(index, files) for g in gs]


def fetch(path: str, rg: int, wanted: list[tuple[str, str]]) -> list[tuple[str, str, object]]:
    """[(video id, class, loudest 3 s at 16 kHz or None)] of the wanted clips of one row group."""
    import numpy as np
    import pyarrow.parquet as pq
    import resampy
    import soundfile as sf
    from building.data.audioset import AudioSetConfig, loudest_window
    cfg = AudioSetConfig()
    with hf_fs().open(path, block_size=1 << 20) as f:
        table = pq.ParquetFile(f).read_row_group(rg, columns=["video_id", "audio"])
    audio = dict(zip(table.column("video_id").to_pylist(), (a["bytes"] for a in table.column("audio").to_pylist())))
    out = []
    for ytid, cls in wanted:
        try:
            x, sr = sf.read(io.BytesIO(audio[ytid]), dtype="float32", always_2d=True)
        except Exception:
            continue
        x = resampy.resample(x.mean(axis=1), sr, cfg.target_sample_rate).astype(np.float32)
        out.append((ytid, cls, loudest_window(x, cfg.target_sample_rate, cfg.target_clip_duration, cfg.window_hop_s, cfg.min_rms)))
    return out


def download_audioset():
    import soundfile as sf
    from building.data.audioset import FOCUS_CLASSES, AudioSetConfig, iter_segments, load_mid_to_focus, slugify
    cfg = AudioSetConfig()
    have = Counter({c: len(list((cfg.processed_dir / slugify(c)).glob("*.wav"))) for c in FOCUS_CLASSES})
    on_disk = {p.stem for p in cfg.processed_dir.glob("*/*.wav")}
    focus = load_mid_to_focus()
    candidates, sizes = defaultdict(list), {}
    for hf_split, csv_split in HF_SPLITS.items():  # each focus segment of the official CSV, located in the HF copy
        where = {}
        for g in row_groups(hf_split):
            key = f"{g['path']}#{g['rg']}"
            sizes[key] = g["bytes"]
            where.update({y: key for y in g["ids"]})
        for ytid, _, _, cls in iter_segments(csv_split, focus):
            if ytid not in on_disk and ytid in where:
                candidates[where[ytid]].append((ytid, cls))
    available = Counter(c for rows in candidates.values() for _, c in rows) + have
    quota, total = {}, 0
    for c in FOCUS_CLASSES:  # PolyChirp: classes in order, up to AUDIOSET_PER_CLASS each, until AUDIOSET_TOTAL
        quota[c] = max(0, min(AUDIOSET_PER_CLASS, available[c], AUDIOSET_TOTAL - total))
        total += quota[c]
    need = Counter({c: quota[c] - have[c] for c in FOCUS_CLASSES})
    gain = lambda key: sum(min(n, max(0, need[c])) for c, n in Counter(c for _, c in candidates[key]).items())  # noqa: E731
    heap = [(-gain(k), k) for k in candidates]
    heapq.heapify(heap)
    spent, written = 0, 0
    with ProcessPoolExecutor(4, mp_context=mp.get_context("spawn")) as ex:
        inflight = {}
        while heap or inflight:
            while heap and len(inflight) < 4 and spent < AUDIOSET_MAX_GB * 1e9:
                _, key = heapq.heappop(heap)
                g = gain(key)
                if g <= 0:
                    continue
                if heap and g < -heap[0][0]:  # lazy greedy: gains only decrease
                    heapq.heappush(heap, (-g, key))
                    continue
                rows = [(y, c) for y, c in candidates[key] if need[c] > 0]
                for c, n in Counter(c for _, c in rows).items():
                    need[c] -= n  # reserved, given back below if the clips fail
                spent += sizes[key]
                path, rg = key.rsplit("#", 1)
                inflight[ex.submit(fetch, path, int(rg), rows)] = Counter(c for _, c in rows)
            if not inflight:
                break
            done, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in done:
                reserved, got = inflight.pop(fut), Counter()
                results = fut.result()
                random.Random(len(results)).shuffle(results)
                for ytid, cls, window in results:
                    if window is None or got[cls] >= reserved[cls]:
                        continue
                    (cfg.processed_dir / slugify(cls)).mkdir(parents=True, exist_ok=True)
                    sf.write(str(cfg.processed_dir / slugify(cls) / f"{ytid}.wav"), window, cfg.target_sample_rate, subtype="PCM_16")
                    got[cls] += 1
                need.update({c: reserved[c] - got[c] for c in reserved})  # give back what failed
                written += sum(got.values())
                print(f"AudioSet: {sum(have.values()) + written}/{sum(quota.values())} clips, {spent / 1e9:.0f} GB read", flush=True)


def main():
    paths.link_polychirp(COLLECTION)
    from building.data.dataset import build_task_dataset
    write_manifest()
    asyncio.run(download_birds())
    download_audioset()
    print(build_task_dataset(COLLECTION, target_species=TARGET_SPECIES, non_target_species=OTHER_SPECIES,
                             non_target_per_species_cap=OTHER_CAP))


if __name__ == "__main__":
    main()
