# tiny-streaming-conformers

Code and results of the paper "Tiny streaming Conformers for real-time bird sound detection on
low-resource acoustic recorders" (submitted to ESANN 2027): a streaming Conformer-like ViT that decides
every 64 ms on the AudioMoth (Cortex-M4F, 48 MHz, 32 KB RAM), compared with PolyChirp's models and
WrenNet on bird species classification and on bird / no-bird detection in long recordings.

## Contents

| File | What it does |
|---|---|
| `models.yaml` | every model, experiment and the training protocol |
| `.env`, `paths.py`, `env.sh` | where the data, runs and virtualenv go (default: inside the repository); `source env.sh` passes uv's folders to uv |
| `vit.py` | our ViT (PyTorch), nothing else |
| `baselines.py` | the baselines: PolyChirp's Mel-PolyChirp ×2 / ×3 (and our changes to PolyChirp), WrenNet's original code |
| `prepare_classification.py` | builds PolyChirp's Paris dataset |
| `prepare_detection.py` | downloads and labels the SSW soundscapes |
| `train.py` | trains any model of any experiment |
| `evaluate.py` | runs → CSV (`results/*_runs.csv`, `results/costs.csv`) |
| `aggregate.py` | CSV → mean ± standard error per model (`results/*_summary.csv`), paired comparisons (`results/*_paired.csv`) |
| `plot.py` | figures (`results/figures/`) |
| `export.py` | INT8 export of the ViT to C, host inference, AudioMoth firmware |
| `audiomoth.py` | the firmware's CSV files (SD card) → latency measured on the AudioMoth (and RAM), into `results/costs.csv` |
| `firmware/` | the ViT and front end in C; AudioMoth, CMSIS as submodules |
| `polychirp/` | PolyChirp (submodule, never modified) |
| `wrennet/` | WrenNet, its authors' code (submodule, never modified) |

`results/` holds the CSV and figures of the paper. `runs/` (weights, scores) and `data/` are not
versioned; `.env` can move them out of the repository ([Where the data go](#where-the-data-go)).

## Setup

```sh
git clone --recursive git@github.com:albento/tiny-streaming-conformers.git
cd tiny-streaming-conformers
source env.sh                    # only if .env moves uv's folders (Where the data go)
uv sync                          # .venv: Python 3.12, PyTorch, TensorFlow 2.16, PolyChirp
source .venv/bin/activate        # or prefix every command with `uv run`
```

- GPU: tested on an NVIDIA L40S; needs an NVIDIA driver ≥ 580 (CUDA 13). Each framework falls back
  to the CPU, with a message, on GPUs it has no kernels for: PolyChirp's TensorFlow 2.16 on Blackwell
  (compute capability 12), our ViT's PyTorch on V100 and older (below 7.5; on a V100, see
  [PyTorch on a V100](#pytorch-on-a-v100)).
- `gcc` (host inference of the INT8 ViT) and, for the firmware and the RAM figures,
  `arm-none-eabi-gcc` on the `PATH`.
- `XC_API_KEY=<your Xeno-Canto API key>` in `polychirp/.env` (classification data only).

### Where the data go

`.env` sets the large folders; an empty variable keeps the default, inside the repository. Fill it
before preparing the data, e.g. on a cluster whose home is small:

```sh
DATA_DIR=/scratch/me/tsc/data                # detection data, ~64 GB (default data/)
RUNS_DIR=/scratch/me/tsc/runs                # runs, a few GB (default runs/)
POLYCHIRP_DATA_DIR=/scratch/me/tsc/polychirp # classification data and its cache, ~22 GB (default inside polychirp/)
```

PolyChirp's code only reads its own folders, so `polychirp/raw_dataset/`,
`polychirp/datasets/scale_49_2_r50/` and its decoded-audio cache (`polychirp/.cache/dataset_cache/`)
become links to `POLYCHIRP_DATA_DIR`. Give each variable a
folder of its own: `train.py --clean` deletes the runs in `RUNS_DIR`.

uv's own large folders are in `.env` too, but uv reads them from the environment only: `source env.sh`
exports the variables of `.env` that are set, in each new shell and job script, before calling uv.

```sh
UV_PROJECT_ENVIRONMENT=/scratch/me/tsc/venv  # the virtualenv, ~12 GB (default .venv/)
UV_CACHE_DIR=/scratch/me/uv-cache            # uv's download cache, ~12 GB (default ~/.cache/uv)
UV_PYTHON_INSTALL_DIR=/scratch/me/uv-python  # the Pythons uv installs (default ~/.local/share/uv/python)
```

```sh
source env.sh                    # then, as usual:
uv sync                          # into UV_PROJECT_ENVIRONMENT
uv run train.py ...              # or source $UV_PROJECT_ENVIRONMENT/bin/activate
```

Keep the cache on the virtualenv's file system: uv then hard-links the packages instead of copying
them. Moving an existing virtualenv breaks it: delete it and run `uv sync` again.

### PyTorch on a V100

The locked PyTorch (CUDA 13) has no kernels for the V100 (compute capability 7.0): the PyTorch
models would train on the CPU. PyTorch 2.14.1 for CUDA 12.6 has them, and gives the same outputs and
gradients as the locked version. Once, after `uv sync`:

```sh
uv pip install -p "${UV_PROJECT_ENVIRONMENT:-.venv}" torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
```

(`--index-url` alone: an `--extra-index-url` of PyPI would take precedence and bring back a CUDA 13
build.) Then `UV_NO_SYNC=1` in `.env`, so that `uv run` keeps this PyTorch instead of the locked one;
after a `git pull` that changes `uv.lock`, run `uv sync` and the line above again. Check:
`uv run python -c "import torch; print(torch.cuda.get_arch_list())"` lists `sm_70`.

This virtualenv is for the PyTorch models only (our ViTs, WrenNet): PyTorch 2.14.1 brings cuDNN 9,
which replaces TensorFlow 2.16's cuDNN 8.9, so PolyChirp's models cannot use the GPU there. For the
PyTorch models, TensorFlow only reads the data, GPU hidden.

## Data

```sh
uv run prepare_classification.py     # PolyChirp's Paris dataset, into polychirp/ (git-ignored folders)
uv run prepare_detection.py          # SSW: ~26 GB downloaded, ~57 GB on disk, into data/
```

Both resume where they stopped.

Data already prepared on another machine can be copied instead (`-L`: PolyChirp's dataset is links
into `raw_dataset/`; the FLAC files are only needed by `prepare_detection.py`):

```sh
rsync -aL --mkpath --exclude audio/ data/ <host>:<DATA_DIR>/
rsync -aL --mkpath polychirp/datasets/scale_49_2_r50 <host>:<POLYCHIRP_DATA_DIR>/datasets/
rsync -aL --mkpath polychirp/raw_dataset/audioset <host>:<POLYCHIRP_DATA_DIR>/raw_dataset/   # augmentation noise
```

**Classification**: PolyChirp's dataset of 3 s clips at 16 kHz around Paris, built by PolyChirp's own
code, with 10 target species (the paper's, fixed) and a `non_target` class (23 other species of the
area, BirdNET-mined clips without birds, AudioSet ambient sounds). Changes to PolyChirp's notebook:
Xeno-Canto only (the Macaulay Library is behind a bot check) and AudioSet from its Hugging Face copy
(YouTube blocks downloads). Xeno-Canto changes over time: a rebuild has the same species and budgets,
not the same clips.

**Detection**: the Cornell Lab's fully annotated soundscapes (Kahl et al., Zenodo, CC BY 4.0), part
of BirdSet. SSW (Sapsucker Woods, 285 h) is split by recording day into train / val / test (≈ 70 /
15 / 15 %, seed 42).

### Labels

One label per 64 ms buffer: bird, no bird, or uncertain (left out of training and metrics). The
annotators drew boxes (time × frequency) around calls but could merge calls of a species less than
5 s apart, so a box often spans silent gaps. Inside each box, `prepare_detection.py` keeps as bird
only the 10 ms frames where the box's band rises above the local noise:

1. spectrogram: 512-point STFT, 10 ms hop, power averaged over 50 ms × 3 bins, in dB;
2. noise floor per frequency bin: the median over the frames outside every box within ±30 s;
3. per frame of a box: SNR = mean of the 3 strongest bins of the box's band (clipped to 150 Hz –
   8 kHz) minus the floor; bird when SNR ≥ 6 dB;
4. clean-up: runs < 50 ms dropped, gaps < 200 ms filled, runs widened by 50 ms, the box's best frame
   always kept; boxes < 1 s kept whole;
5. a buffer is bird if a bird frame falls in it, else uncertain if a box covers it, else no bird.

SSW: 18.4 % bird, 4.8 % uncertain. Inspect any stretch with
`uv run plot.py labels <recording> --start <s> --end <s>`.

## Models

Our ViT, `vit_<stem>_l<layers>[_nc]`: learned PCEN front end (STFT per 64 ms buffer), a convolutional
stem, then per layer a Conformer-style convolution module (depthwise, kernel 7) and a Transformer
encoder layer with ALiBi attention (dim 32, 4 heads, FFN 128).

| Option | Values |
|---|---|
| stem | `s24`: 5×5, stride 2×4 (a token every ~37 ms); `s27`: 5×7, stride 2×7 (a token per 64 ms buffer) |
| layers | 1, 2, 3 |
| `_nc` | non-causal (float reference); default causal: streams on the AudioMoth, attention on the last 32 tokens |

The baselines: PolyChirp's `mel_polychirp_x2` and `mel_polychirp_x3`, on the front end PolyChirp's code
assigns them, and `wrennet`, its authors' PyTorch code (submodule `wrennet/`: their final configuration,
learned log-linear front end at 32 kHz, GRU, 53.6k parameters), trained by `train.py` like the ViT, on
our 16 kHz data upsampled to 32 kHz; float only. Its front end's filter-bank cache froze the learned
front end after the first epoch: fixed in `baselines.wrennet`. PolyChirp's port of WrenNet (no GRU,
its own mel front end), used before, scored 0.006 macro F2 below it.

| Experiment | Task | Models |
|---|---|---|
| `classification` | 10 species, 3 s clips (PolyChirp's benchmark) | 12 ViTs + 3 baselines |
| `detection_stream` | bird / no bird every 64 ms, streamed as on the device (INT8) | 6 causal ViTs |
| `detection_window` | bird / no bird in each 3 s window, a decision every 1.47 s (all models in the clip models' regime) | 12 ViTs + 3 baselines |

Protocol, every model: Adam (lr 1e-3), batch 32, the checkpoint with the best validation macro
average precision (min delta 1e-4), learning rate halved after 3 epochs without gain, stop after 10,
at most 100 epochs. Classification uses PolyChirp's data pipeline (clips, augmentation, class weights)
for every model, and PolyChirp's own training for its models. Detection draws the same 30 s regions,
decision times and window order for every model of a seed.

## Training

10 runs per model, seeds 0–9. A run goes to `runs/<experiment>/<model>/seed_<n>/`; finished runs are
skipped, so any command resumes.

```sh
uv run train.py --all                     # everything (360 runs)
uv run train.py --all --parallel 4        # 4 runs at a time on one GPU (each ViT run needs < 4 GB)
uv run train.py classification --runs 10  # one experiment
uv run train.py --list                    # what is done
uv run train.py --clean                   # delete the runs and results/, to start again
```

One model at a time:

```sh
# classification
uv run train.py classification vit_s24_l1 --runs 10
uv run train.py classification vit_s24_l2 --runs 10
uv run train.py classification vit_s24_l3 --runs 10
uv run train.py classification vit_s27_l1 --runs 10
uv run train.py classification vit_s27_l2 --runs 10
uv run train.py classification vit_s27_l3 --runs 10
uv run train.py classification vit_s24_l1_nc --runs 10
uv run train.py classification vit_s24_l2_nc --runs 10
uv run train.py classification vit_s24_l3_nc --runs 10
uv run train.py classification vit_s27_l1_nc --runs 10
uv run train.py classification vit_s27_l2_nc --runs 10
uv run train.py classification vit_s27_l3_nc --runs 10
uv run train.py classification mel_polychirp_x2 --runs 10
uv run train.py classification mel_polychirp_x3 --runs 10
uv run train.py classification wrennet --runs 10
# detection_stream
uv run train.py detection_stream vit_s24_l1 --runs 10
uv run train.py detection_stream vit_s24_l2 --runs 10
uv run train.py detection_stream vit_s24_l3 --runs 10
uv run train.py detection_stream vit_s27_l1 --runs 10
uv run train.py detection_stream vit_s27_l2 --runs 10
uv run train.py detection_stream vit_s27_l3 --runs 10
# detection_window
uv run train.py detection_window vit_s24_l1 --runs 10
uv run train.py detection_window vit_s24_l2 --runs 10
uv run train.py detection_window vit_s24_l3 --runs 10
uv run train.py detection_window vit_s27_l1 --runs 10
uv run train.py detection_window vit_s27_l2 --runs 10
uv run train.py detection_window vit_s27_l3 --runs 10
uv run train.py detection_window vit_s24_l1_nc --runs 10
uv run train.py detection_window vit_s24_l2_nc --runs 10
uv run train.py detection_window vit_s24_l3_nc --runs 10
uv run train.py detection_window vit_s27_l1_nc --runs 10
uv run train.py detection_window vit_s27_l2_nc --runs 10
uv run train.py detection_window vit_s27_l3_nc --runs 10
uv run train.py detection_window mel_polychirp_x2 --runs 10
uv run train.py detection_window mel_polychirp_x3 --runs 10
uv run train.py detection_window wrennet --runs 10
```

## Evaluation

```sh
uv run evaluate.py      # results/classification_runs.csv, detection_runs.csv, costs.csv
uv run aggregate.py     # results/{classification,detection}_{summary,paired}.csv
uv run plot.py          # results/figures/
```

`*_paired.csv` compares every pair of models seed by seed (`diff_mean` = model_b − model_a, its 95 %
confidence interval `diff_ci95`, the paired t-test's `p` and its Holm correction `p_holm` over the pairs
of the same experiment, test set, float / INT8 and metric). The runs of two models share their seeds,
so two models can differ clearly although the confidence intervals of their means overlap.

`evaluate.py` is incremental: it adds the finished runs not in the CSV yet and keeps the rows already
there, so runs evaluated on other machines can be gathered in one CSV (`--redo` evaluates again the
runs present). The curves need each run's scores (`scores.npz` or `results.npz`, `curves.npz`) and
`train.json`, not its weights.

- Classification (test clips): macro F2 at per-class thresholds tuned on train + val (PolyChirp's
  evaluation), macro average precision, clip error (predicted set of species ≠ true set).
- Detection (SSW test days), the first 8 s of every recording left out:
  - segment ROC (AUC, AP) on common 1.47 s segments, positive when they overlap an annotation box; a
    model's score combines its decisions judging the segment: the max of the 23 buffers' (streaming;
    each judges a part of it), the mean of the two 3 s windows containing it (each judges all of it);
  - the AudioMoth's triggered recording simulated on the scores (a recording starts with a
    pre-roll and ends 1.5 s after the last detection): time recorded vs annotation boxes captured,
    summarized as the time needed for 80 / 90 % of the boxes and the boxes captured with 20 / 30 % of
    the time.
- Float and INT8 for every model but WrenNet (float only): the ViT's INT8 is its firmware C run on the
  host (`firmware/host_infer.c`), PolyChirp's models' their INT8 TFLite model.
- Costs: parameters, multiply-accumulates per second of audio (front end included), flash, RAM.
- Summaries: mean over runs ± standard error (standard deviation / √runs).

## AudioMoth firmware

```sh
uv run export.py runs/classification/vit_s24_l3/seed_0     # -> audiomoth_vit_s24_l3_seed_0.bin in that folder
uv run export.py --all                                     # every causal ViT (detection_stream, seed 0) -> runs/audiomoth/
uv run audiomoth.py <folder of the SD cards' CSV files>    # -> results/audiomoth.csv, ram_kb and latency_max_ms of results/costs.csv
```

The latency-measuring firmware: the official AudioMoth firmware (`firmware/AudioMoth-Firmware-Basic`,
built with `firmware/AudioMoth-Project`) with the ViT added by `firmware/main.patch`. Flash it with the
AudioMoth Flash App and let it record (switch on CUSTOM, a schedule set with the AudioMoth Configuration
App, or DEFAULT): each recording period runs the ViT on the microphone, at 16 kHz and at the full core
clock whatever the configuration (sample rate and energy saver mode are ignored), and writes, instead of the WAV file, a CSV file
with the cycles of the front end and of the ViT for every 64 ms buffer, the buffers skipped and the
deepest stack. `audiomoth.py` reads them (any folder layout; each file names its model): the max and
mean latency per buffer, real time when the max is under 64 ms and no buffer was skipped, and the
deepest stack (kept in `results/audiomoth.csv`; the RAM of `costs.csv` leaves the stack out).
`evaluate.py --redo` keeps these values.

The EFM32 has 32 KB of RAM for the official firmware's state (about 19 KB), the ViT's and the front
end's, a 2 KB heap and a 1 KB stack. When a model does not fit (the linker's "RAM memory overflowed!",
the 3-layer ViTs), `export.py` builds it again with `-DVIT_NO_FREQUENCY_TRIGGER` (`firmware/main.patch`):
without the 4 KB Hamming window of the official firmware's frequency trigger, which only `makeRecording()`
uses and this firmware never calls. That `.bin` is named `..._no_freq_trigger.bin`.

## Reproducibility

Seeds fix the data order, the initialisation and the augmentation; some GPU kernels are not
deterministic, so a rerun matches closely, not bit for bit.
