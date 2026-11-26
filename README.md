# tiny-streaming-conformers

Code and results of the paper "Tiny Streaming Conformers for Real-Time Bioacoustic Monitoring on
Resource-Constrained Microcontrollers" (submitted to ESANN 2027): a streaming Conformer that decides
every 64ms on the AudioMoth (Cortex-M4F, 48MHz, 32KB RAM), compared with PolyChirp's models and
WrenNet on bird species classification and on bird / no-bird detection in long recordings.

## Contents

| File | What it does |
|---|---|
| `models.yaml` | every model, experiment and the training protocol |
| `.env`, `paths.py` | where the data and runs go (default: inside the repository) |
| `conformer.py` | our Conformer (PyTorch), nothing else |
| `baselines.py` | the baselines: PolyChirp's Mel-PolyChirp ×2 / ×3 (and our changes to PolyChirp), WrenNet's original code |
| `prepare_classification.py` | builds PolyChirp's Paris dataset |
| `prepare_detection.py` | downloads and labels the SSW soundscapes |
| `train.py` | trains any model of any experiment |
| `evaluate.py` | runs → CSV (`results/*_runs.csv`, `results/costs.csv`) |
| `aggregate.py` | CSV → mean ± standard error per model (`results/*_summary.csv`), paired comparisons (`results/*_paired.csv`) |
| `plot.py` | figures (`results/figures/`) |
| `export.py` | INT8 export of the Conformer to C, host inference, AudioMoth firmware |
| `audiomoth.py` | the firmware's CSV files (SD card) → latency measured on the AudioMoth (and RAM), into `results/costs.csv` |
| `firmware/` | the Conformer and front end in C, the AudioMoth firmware's files we change (`audiomoth/`); AudioMoth, CMSIS as submodules |
| `polychirp/` | PolyChirp (submodule, never modified) |
| `wrennet/` | WrenNet, its authors' code (submodule, never modified) |

`results/` holds the CSV and figures of the paper. `runs/` (weights, scores) and `data/` are not
versioned; `.env` can move them out of the repository ([Where the data go](#where-the-data-go)).

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```sh
git clone --recursive git@github.com:albento/tiny-streaming-conformers.git
cd tiny-streaming-conformers
uv sync                          # .venv: Python 3.12, PyTorch, TensorFlow 2.16, PolyChirp
source .venv/bin/activate        # or prefix every command with `uv run`
```

For the classification data, put your Xeno-Canto API key in `polychirp/.env`:
`XC_API_KEY=<your key>`.

`gcc`, `make` and the Arm embedded toolchain (`arm-none-eabi-gcc`) must be on the `PATH`:

```sh
sudo apt install gcc make gcc-arm-none-eabi                      # Debian, Ubuntu
sudo pacman -S gcc make arm-none-eabi-gcc arm-none-eabi-newlib   # Arch
brew install --cask gcc-arm-embedded                             # macOS (gcc, make: Xcode's command line tools)
```

### Where the data go

`.env` sets the large folders; an empty variable keeps the default, inside the repository. Fill it
before preparing the data:

```sh
CLASSIFICATION_DATA_DIR=/path/to/classification # classification data and its cache, ~22GB (default inside polychirp/)
DETECTION_DATA_DIR=/path/to/detection           # detection data, ~64GB (default data/)
RUNS_DIR=/path/to/runs                          # runs, a few GB (default runs/)
```

PolyChirp's code only reads its own folders, so `polychirp/raw_dataset/`,
`polychirp/datasets/scale_49_2_r50/` and its decoded-audio cache (`polychirp/.cache/dataset_cache/`)
become links to `CLASSIFICATION_DATA_DIR`. Give each variable a
folder of its own: `train.py --clean` deletes the runs in `RUNS_DIR`.

## Data

```sh
uv run prepare_classification.py     # PolyChirp's Paris dataset, into polychirp/ (git-ignored folders)
uv run prepare_detection.py          # SSW: ~26GB downloaded, ~57GB on disk, into data/
```

**Classification**: PolyChirp's dataset of 3s clips at 16kHz around Paris, built by PolyChirp's own
code, with 10 target species (the paper's, fixed) and a `non_target` class (23 other species of the
area, BirdNET-mined clips without birds, AudioSet ambient sounds). Changes to PolyChirp's notebook:
Xeno-Canto only (the Macaulay Library is behind a bot check) and AudioSet from its Hugging Face copy. Xeno-Canto changes over time: a rebuild has the same species and budgets,
not the same clips.

**Detection**: the Cornell Lab's fully annotated soundscapes (Kahl et al., Zenodo, CC BY 4.0), part
of BirdSet. SSW (Sapsucker Woods, 285h) is split by recording day into train / val / test (≈ 70 /
15 / 15%, seed 42).

## Models

Our Conformer, `conf_<stem>_l<layers>[_nc]`: learned PCEN front end (STFT per 64ms buffer), a convolutional
stem, then per layer a Conformer-style convolution module (depthwise, kernel 7) and a Transformer
encoder layer with ALiBi attention (dim 32, 4 heads, FFN 128).

| Option | Values |
|---|---|
| stem | `s24`: 5×5, stride 2×4 (a token every ~37ms); `s27`: 5×7, stride 2×7 (a token per 64ms buffer) |
| layers | 1, 2, 3 |
| `_nc` | non-causal (float reference); default causal: streams on the AudioMoth, attention on the last 32 tokens |

The baselines: PolyChirp's `mel_polychirp_x2` and `mel_polychirp_x3`, on the front end PolyChirp's code
assigns them, and `wrennet`, its authors' PyTorch code (submodule `wrennet/`: their final configuration,
learned log-linear front end at 32kHz, GRU, 53.6k parameters), trained by `train.py` like the Conformer, on
our 16kHz data upsampled to 32kHz; float only. Its front end's filter-bank cache froze the learned
front end after the first epoch: fixed in `baselines.wrennet`. PolyChirp's port of WrenNet (no GRU,
its own mel front end), used before, scored 0.006 macro F2 below it.

| Experiment | Task | Models |
|---|---|---|
| `classification` | 10 species, 3s clips (PolyChirp's benchmark) | 12 Conformers + 3 baselines |
| `detection_stream` | bird / no bird every 64ms, streamed as on the device (INT8) | 6 causal Conformers |
| `detection_window` | bird / no bird in each 3s window, a decision every 1.47s (all models in the clip models' regime) | 12 Conformers + 3 baselines |

Protocol, every model: Adam (lr 1e-3), batch 32, the checkpoint with the best validation macro
average precision (min delta 1e-4), learning rate halved after 3 epochs without gain, stop after 10,
at most 100 epochs. Classification uses PolyChirp's data pipeline (clips, augmentation, class weights)
for every model, and PolyChirp's own training for its models. Detection draws the same 30s regions,
decision times and window order for every model of a seed.

## Training

5 runs per model, seeds 0–4. A run goes to `runs/<experiment>/<model>/seed_<n>/`; finished runs are
skipped, so any command resumes.

```sh
uv run train.py --all                                # everything (360 runs)
uv run train.py --all --parallel 4                   # 4 runs at a time on one GPU (each Conformer run needs < 4GB)
uv run train.py classification --runs 5              # one experiment
uv run train.py classification conf_s27_l2 --runs 5  # one model of one experiment
uv run train.py --list                               # what is done
uv run train.py --clean                              # delete the runs and results/, to start again
```

## Evaluation

```sh
uv run evaluate.py      # results/classification_runs.csv, detection_runs.csv, costs.csv
uv run aggregate.py     # results/{classification,detection}_{summary,paired}.csv
uv run plot.py          # results/figures/
```

`*_paired.csv` compares every pair of models seed by seed (`diff_mean` = model_b − model_a, its 95%
confidence interval `diff_ci95` and the test's p-value).

`evaluate.py` is incremental: it adds the finished runs not in the CSV yet and keeps the rows already
there (`--redo` evaluates again the runs present). The curves need each run's scores (`scores.npz` or `results.npz`, `curves.npz`) and
`train.json`, not its weights.

- Classification (test clips): macro F2 at per-class thresholds tuned on train + val (PolyChirp's
  evaluation), macro average precision, clip error (predicted set of species ≠ true set).
- Detection (SSW test days), the first 8s of every recording left out:
  - segment ROC (AUC, AP) on common 1.47s segments, positive when they overlap an annotation box; a
    model's score combines its decisions judging the segment: the max of the 23 buffers' (streaming;
    each judges a part of it), the mean of the two 3s windows containing it (each judges all of it);
  - the AudioMoth's triggered recording simulated on the scores (a recording starts with a
    pre-roll and ends 1.5s after the last detection): time recorded vs annotation boxes captured,
    summarized as the time needed for 80 / 90% of the boxes and the boxes captured with 20 / 30% of
    the time.
- Float for every model; INT8 also in classification (causal Conformers and PolyChirp's models) and
  detection_stream: the Conformer's INT8 is its firmware C run on the host (`firmware/host_infer.c`),
  PolyChirp's models' their INT8 TFLite model. WrenNet and detection_window: float only.
- Costs: parameters, multiply-accumulates per second of audio (front end included), flash, RAM.
- Summaries: mean over runs ± standard error (standard deviation / √runs).

## AudioMoth firmware

```sh
uv run export.py runs/classification/conf_s24_l3/seed_0    # -> audiomoth_conf_s24_l3_seed_0.bin in that folder
uv run export.py --all                                     # every causal Conformer (detection_stream, seed 0) -> runs/audiomoth/
uv run audiomoth.py <folder of the SD cards' CSV files>    # -> results/audiomoth.csv, ram_kb and latency_max_ms of results/costs.csv
```

The latency-measuring firmware: the official AudioMoth firmware (`firmware/AudioMoth-Firmware-Basic`,
built with `firmware/AudioMoth-Project`) whose `main.c`, `digitalfilter.c` and `Makefile` are replaced by
those of `firmware/audiomoth/`, which add the Conformer. Flash it with the
AudioMoth Flash App and let it record (switch on CUSTOM, a schedule set with the AudioMoth Configuration
App, or DEFAULT): each recording period runs the Conformer on the microphone, at 16kHz and at the full core
clock whatever the configuration (sample rate and energy saver mode are ignored), and writes, instead of the WAV file, a CSV file
with the cycles of the front end and of the Conformer for every 64ms buffer, the buffers skipped and the
deepest stack. `audiomoth.py` reads them (any folder layout; each file names its model): the max and
mean latency per buffer, real time when the max is under 64ms and no buffer was skipped, and the
deepest stack (kept in `results/audiomoth.csv`; the RAM of `costs.csv` leaves the stack out).
`evaluate.py --redo` keeps these values.

The EFM32 has 32KB of RAM for the official firmware's state (about 19KB), the Conformer's and the front
end's, a 2KB heap and a 1KB stack. When a model does not fit (the linker's "RAM memory overflowed!",
the 3-layer Conformers), `export.py` builds it again with `-DCONF_NO_FREQUENCY_TRIGGER` (`firmware/audiomoth/digitalfilter.c`):
without the 4KB Hamming window of the official firmware's frequency trigger, which only `makeRecording()`
uses and this firmware never calls. That `.bin` is named `..._no_freq_trigger.bin`.

## Reproducibility

Seeds fix the data order, the initialisation and the augmentation; some GPU kernels are not
deterministic, so a rerun matches closely, not bit for bit.
