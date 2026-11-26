"""The baselines: PolyChirp's models, as PolyChirp builds and trains them, and WrenNet, as its
authors implemented it.

Mel-PolyChirp x2 / x3: PolyChirp's mel_ultralight, on its MEL_ULTRALIGHT front end, with our changes
to PolyChirp, applied at run time (the submodule is never modified):
- the training protocol: PolyChirp's runner keeps the weights of the epoch with the best validation
  loss and stops on it; here, as for every model of the paper, on the best validation macro average
  precision (min_delta 1e-4, at most 100 epochs, patience 10; the learning-rate plateau follows it);
- the run folders: under runs/ instead of the submodule's models/.

WrenNet: its authors' PyTorch code (submodule wrennet/, unmodified), trained by train.py like the Conformer
(PolyChirp's port of it, without the GRU, falls 0.006 macro F2 short of it).
"""
import inspect
import sys
import types
from pathlib import Path

import numpy as np

import paths

COLLECTION = "scale_49_2_r50"  # the Paris dataset (prepare_classification.py)
paths.link_polychirp(COLLECTION)
K = 10                         # the 10 target species; non_target is the all-zeros negative


def specs() -> dict:
    from building.models import TinyChirpModel, mel_cnn_new
    return {"mel_polychirp_x2": TinyChirpModel(base_model=mel_cnn_new.ULTRALIGHT, multi=2.0),
            "mel_polychirp_x3": TinyChirpModel(base_model=mel_cnn_new.ULTRALIGHT, multi=3.0)}


def species(catalog, seed: int) -> tuple[list[int], int]:
    """(the target classes' indices, the non_target index), drawn as PolyChirp's run_scale does for
    run `seed` (k = 10 of the 10 species: only their order depends on the seed)."""
    pool = [i for i, n in enumerate(catalog.class_names) if n != "non_target"]
    return np.random.default_rng([seed, K, 0]).choice(pool, K, replace=False).tolist(), catalog.class_names.index("non_target")



# --- classification: PolyChirp's own training, with the paper's protocol ---------------------------

def install_protocol(runner, log) -> None:
    """Replaces, in PolyChirp's runner.train_cell, the compared validation loss by minus the
    validation macro AP, and its epoch line by log (train.py's: losses, AP, learning rate, time).
    The source is checked, so that a change upstream fails loudly."""
    from sklearn.metrics import average_precision_score

    def val_macro_ap(batches):
        y = np.concatenate([b[0] for b in batches])
        return float(average_precision_score(y, np.concatenate([b[1] for b in batches]), average="macro"))
    src = inspect.getsource(runner.train_cell)
    for old, new in (
        ('        cell_tag = f"[scale] s{sample_idx} k{k}"\n',
         '        cell_tag = f"[scale] s{sample_idx} k{k}"\n        best_epochs: dict = {}\n'),
        ("            t_train = perf_counter()\n",
         "            t_train = perf_counter()\n            train_losses: dict = {}\n            epoch_log: dict = {}\n"),
        ("                    s.model.train_on_batch(feats[s.spec.input_frontend], y, sample_weight=sw)\n",
         "                    train_losses.setdefault(id(s), []).append(float(np.ravel(s.model.train_on_batch(feats[s.spec.input_frontend], y, sample_weight=sw))[0]))\n"),
        ("            val_losses: dict[int, list[float]] = {id(s): [] for s in active}\n",
         "            val_losses: dict[int, list[float]] = {id(s): [] for s in active}\n            val_preds: dict = {}\n"),
        ("                    out = s.model.test_on_batch(feats[s.spec.input_frontend], y, sample_weight=sw)\n",
         "                    out = s.model.test_on_batch(feats[s.spec.input_frontend], y, sample_weight=sw)\n"
         "                    val_preds.setdefault(id(s), []).append((np.asarray(y), np.asarray(s.model.predict_on_batch(feats[s.spec.input_frontend]))))\n"),
        ("                vloss = float(np.mean(val_losses[id(s)])) if val_losses[id(s)] else math.inf\n",
         "                vloss = -val_macro_ap(val_preds[id(s)])  # minimized\n"
         "                epoch_log[id(s)] = (float(np.mean(train_losses[id(s)])), float(np.mean(val_losses[id(s)])), -vloss,\n"
         "                                    float(tf.keras.backend.get_value(s.model.optimizer.learning_rate)))\n"),
        (  # PolyChirp's epoch line -> train.py's
         '            bests = " ".join(\n'
         '                f"{s.spec.name}x{s.spec.multi:g}={s.best_val_loss:.4f}"\n'
         '                + (" [stop]" if s.stopped else "")\n'
         '                for s in active\n'
         '            )\n'
         '            detail = ""\n'
         '            if FIT_VERBOSE:\n'
         '                steps = " ".join(\n'
         '                    f"{s.spec.name}x{s.spec.multi:g}={t_step[id(s)]:.1f}s" for s in active\n'
         '                )\n'
         '                detail = f"data={t_data:.1f}s feat={t_feat:.1f}s {steps} "\n'
         '            tqdm.write(\n'
         '                f"{cell_tag} e{epoch + 1}/{config.epochs} "\n'
         '                f"train={train_elapsed:.1f}s val={val_elapsed:.1f}s {detail}| {bests}"\n'
         '            )\n',
         "            for s in active:\n"
         "                train_loss, val_loss, ap, lr = epoch_log[id(s)]\n"
         "                if s.epochs_no_improve == 0:\n"
         "                    best_epochs[id(s)] = epoch + 1\n"
         "                log(epoch + 1, train_loss, val_loss, ap, -s.best_val_loss, best_epochs[id(s)], lr, train_elapsed + val_elapsed,\n"
         "                    s.epochs_no_improve == 0, float(tf.keras.backend.get_value(s.model.optimizer.learning_rate)),\n"
         "                    s.stopped or epoch + 1 == config.epochs)\n"),
    ):
        assert src.count(old) == 1, f"PolyChirp's train_cell changed: {old.strip()!r}"
        src = src.replace(old, new)
    namespace = dict(vars(runner), val_macro_ap=val_macro_ap, log=log)
    exec(compile(src, inspect.getsourcefile(runner.train_cell), "exec"), namespace)
    runner.train_cell = namespace["train_cell"]


def quiet_export() -> None:
    """PolyChirp's TFLite recipe exports a SavedModel first, and Keras lists its ~200 captured tensors
    in the log: off by default (verbose=False unless asked)."""
    import keras
    if not getattr(keras.Model.export, "quiet", False):
        export = keras.Model.export

        def quiet(self, filepath, format="tf_saved_model", verbose=False, **kwargs):
            return export(self, filepath, format=format, verbose=verbose, **kwargs)
        quiet.quiet = True
        keras.Model.export = quiet


def single_thread_tflite() -> None:
    """PolyChirp scores the INT8 TFLite model clip by clip with os.cpu_count() - 4 threads per run:
    the node's cores, not the job's (a container or SLURM job often gets a few of them), so runs in
    parallel oversubscribe the CPU by 10-100x and a tiny model spends its time synchronizing threads.
    One thread per interpreter: the fastest for a clip-sized model, and what the device does."""
    from building.polychirp_training import results
    if not getattr(results.make_tflite_runner, "single_thread", False):
        make = results.make_tflite_runner

        def runner(tflite_path, num_threads=1):
            return make(tflite_path, 1)
        runner.single_thread = True
        results.make_tflite_runner = runner


def train_classification(name: str, seed: int, run_dir: Path, protocol: dict, log) -> None:
    """One run of PolyChirp's scale benchmark (k = 10, one cell): PolyChirp trains, tunes the per-class
    thresholds, quantizes to INT8 TFLite and writes results.npz (every clip's float and INT8 scores on
    the tuning set, train + val, and on the test set) into run_dir. log(epoch, ...): train.py's epoch line."""
    import keras
    from building.polychirp_training import runner
    install_protocol(runner, log)
    quiet_export()
    single_thread_tflite()
    runner.scale_run_dir = lambda *args: run_dir
    keras.utils.set_random_seed(seed)
    config = runner.ScaleGridConfig(collection=COLLECTION, specs=[specs()[name]], epochs=protocol["epochs"],
                                    patience=protocol["patience"], batch_size=protocol["batch_size"], seed=seed,
                                    min_delta=protocol["min_delta"])
    runner.run_scale(config, n_samples=1, k_values=[K])


# --- detection: a one-output model on 3 s windows --------------------------------------------------

def build_detector(name: str):
    """(Keras model with one sigmoid output, its front end: audio [n, 48000] in [-1, 1] -> features)."""
    from building.polychirp_training.training import featurize
    spec = specs()[name]
    quiet_export()
    return spec.builder(1), lambda audio: featurize(audio, spec.input_frontend)


# --- WrenNet: its authors' code (submodule wrennet/), in PyTorch ---------------------------------------

# Improved_Phi_GRU_ATT as config/experiments/four_birds_combined.yaml sets it (the authors' final
# version, 53,581 parameters at 10 classes): a semi-learnable log-linear filter bank (64 filters, the
# breakpoint and transition width learned) on a 32 kHz STFT, AmplitudeToDB (top_db 80) and per-clip
# standardization, MatchboxNetSkip (its defaults hold: the config's num_layers / kernel_size / dropout
# are not the keys it reads), a GRU, attention pooling. f_min / f_max: what the authors' training script
# passed (the dataset's lowcut / highcut; the refactored code's f_min = 0 makes the log scale NaN).
WRENNET = dict(spectrogram_type="combined_log_linear", sample_rate=32000, n_mel_bins=64, n_linear_filters=64, hidden_dim=64,
               n_fft=1024, hop_length=320, f_min=150.0, f_max=10000.0, breakpoint=4000, transition_width=100,
               matchbox=dict(base_filters=32, num_layers=3, kernel_size=5, stride=1, dropout=0.05))
WRENNET_LR = {"breakpoint": 3.0, "transition_width": 0.1}  # the config's own learning rates of the two front-end parameters


def wrennet_imports() -> None:
    """What WrenNet's code imports beyond PyTorch: torchaudio (for AmplitudeToDB only, rewritten here:
    torchaudio's last release, 2.11, built for CUDA 13.0, does not load with this PyTorch), omegaconf
    (a type hint) and torchsummary (unused); then the submodule, after the repository (both have a train.py)."""
    import torch

    class AmplitudeToDB(torch.nn.Module):
        """torchaudio 2.11's transforms.AmplitudeToDB (amin 1e-10, ref 1), its quirk included: a 3-D input
        [batch, freq, time] gets one top_db cut-off for the whole batch."""

        def __init__(self, stype: str = "power", top_db: float | None = None):
            super().__init__()
            self.multiplier, self.top_db = 10.0 if stype == "power" else 20.0, top_db

        def forward(self, x):
            x_db = self.multiplier * torch.log10(torch.clamp(x, min=1e-10))
            if self.top_db is not None:
                shape = x_db.size()
                x_db = x_db.reshape(-1, shape[-3] if x_db.dim() > 2 else 1, shape[-2], shape[-1])
                x_db = torch.max(x_db, (x_db.amax(dim=(-3, -2, -1)) - self.top_db).view(-1, 1, 1, 1)).reshape(shape)
            return x_db
    modules = {name: types.ModuleType(name) for name in ("torchaudio", "torchaudio.functional", "torchaudio.transforms", "omegaconf", "torchsummary")}
    modules["torchaudio"].functional, modules["torchaudio"].transforms = modules["torchaudio.functional"], modules["torchaudio.transforms"]
    modules["torchaudio.transforms"].AmplitudeToDB = AmplitudeToDB
    modules["omegaconf"].DictConfig = dict
    modules["torchsummary"].summary = None
    for name, module in modules.items():
        sys.modules.setdefault(name, module)
    if str(paths.ROOT / "wrennet") not in sys.path:
        sys.path.append(str(paths.ROOT / "wrennet"))


def wrennet(n_outputs: int):
    """WrenNet (PyTorch), one logit per output, with a fix: its front end caches its filter bank while the
    breakpoint and transition width are unchanged, and the bank cached during validation (no gradient)
    is then reused in training, freezing both after the first epoch: the cache is cleared before every
    forward pass that needs gradients."""
    import torch
    wrennet_imports()
    from birds_distillation_edge.models.legacy_models import Improved_Phi_GRU_ATT
    model = Improved_Phi_GRU_ATT(num_classes=n_outputs, **WRENNET)
    model.combined_log_linear_spec.register_forward_pre_hook(lambda spec, args: spec.rebuild_filters() if torch.is_grad_enabled() else None)
    return model


def wrennet_parameters(model, lr: float) -> list[dict]:
    """The optimizer's parameter groups: the front end's breakpoint and transition width at their own rates."""
    spec = model.combined_log_linear_spec
    own = {"breakpoint": spec.breakpoint, "transition_width": spec.transition_width}
    rest = [p for p in model.parameters() if all(p is not q for q in own.values())]
    return [{"params": rest, "lr": lr}] + [{"params": [p], "lr": WRENNET_LR[name]} for name, p in own.items()]


def wrennet_input(audio) -> np.ndarray:
    """Audio [n, samples] in [-1, 1] at 16 kHz (PolyChirp's clips, the soundscapes), as float32 [n, samples]:
    still 16 kHz, wrennet_forward brings it to WrenNet's 32 kHz on the model's device."""
    return np.asarray(audio, np.float32).reshape(len(audio), -1)


_UPSAMPLE = {}


def upsample2(x):
    """scipy.signal.resample_poly(x, 2, 1, axis=1) on x's device: torch [n, samples] -> [n, 2 samples].
    Its filter (firwin, 41 taps, Kaiser 5, cut-off at the old Nyquist, times 2) after one zero, applied
    to x with a zero after every sample; the output keeps samples 21 .. 21 + 2 samples of the full
    convolution, as scipy's. On the GPU: the CPU no longer slows the training down (scipy, one core,
    each window resampled for every decision it belongs to)."""
    import torch
    import torch.nn.functional as F
    from scipy.signal import firwin
    key = (x.device, x.dtype)
    if key not in _UPSAMPLE:
        h = np.concatenate([[0.0], firwin(41, 0.5, window=("kaiser", 5.0)).astype(np.float32) * 2])
        _UPSAMPLE[key] = torch.from_numpy(h[::-1].copy()).to(x)[None, None]   # conv1d correlates: the flipped filter
    w = _UPSAMPLE[key]
    up = torch.zeros(x.shape[0], 2 * x.shape[1], dtype=x.dtype, device=x.device)
    up[:, ::2] = x
    full = F.conv1d(F.pad(up, (w.shape[-1] - 1, w.shape[-1] - 1))[:, None], w)[:, 0]
    return full[:, 21:21 + up.shape[1]]


def wrennet_forward(model, x):
    """WrenNet's logits for 16 kHz audio x (torch [n, samples] in [-1, 1], on the model's device): 32 kHz first."""
    return model(upsample2(x))
