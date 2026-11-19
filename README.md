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
| `prepare_detection.py` | downloads and labels the SSW / SNE soundscapes |
| `train.py` | trains any model of any experiment |
| `evaluate.py` | runs → CSV (`results/*_runs.csv`, `results/costs.csv`) |
| `aggregate.py` | CSV → mean ± standard error per model (`results/*_summary.csv`), paired comparisons (`results/*_paired.csv`) |
| `plot.py` | figures (`results/figures/`) |
| `export.py` | INT8 export of the ViT to C, host inference, AudioMoth firmware |
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
  (compute capability 12), our ViT's PyTorch on V100 and older (below 7.5).
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

## Data

```sh
uv run prepare_classification.py     # PolyChirp's Paris dataset, into polychirp/ (git-ignored folders)
uv run prepare_detection.py          # SSW + SNE: ~31 GB downloaded, ~60 GB on disk, into data/
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
15 / 15 %, seed 42); SNE (Sierra Nevada, 33 h) is a test set from a site never seen in training.

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

SSW: 18.4 % bird, 4.8 % uncertain; SNE: 33.4 % bird, 16.3 % uncertain. Inspect any stretch with
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
- Detection (SSW test days, SNE), the first 8 s of every recording left out:
  - segment ROC (AUC, AP) on common 1.47 s segments, positive when they overlap an annotation box; a
    model's score is its highest decision judging the segment;
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
```

The latency-measuring firmware: the official AudioMoth firmware (`firmware/AudioMoth-Firmware-Basic`,
built with `firmware/AudioMoth-Project`) with the ViT added by `firmware/main.patch`. Flash it with the
AudioMoth Flash App.

## Reproducibility

Seeds fix the data order, the initialisation and the augmentation; some GPU kernels are not
deterministic, so a rerun matches closely, not bit for bit.
