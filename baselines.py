"""PolyChirp's models, as PolyChirp builds and trains them, and our changes to PolyChirp, applied at
run time (the submodule is never modified):

- the training protocol: PolyChirp's runner keeps the weights of the epoch with the best validation
  loss and stops on it; here, as for every model of the paper, on the best validation macro average
  precision (min_delta 1e-4, at most 100 epochs, patience 10; the learning-rate plateau follows it);
- the run folders: under runs/ instead of the submodule's models/.

Mel-PolyChirp x2 / x3 (PolyChirp's mel_ultralight, on its MEL_ULTRALIGHT front end) and WrenNet
(PolyChirp's port, on MEL_HIGHER): each on the front end PolyChirp's code assigns it.
"""
import inspect
from pathlib import Path

import numpy as np

import paths

COLLECTION = "scale_49_2_r50"  # the Paris dataset (prepare_classification.py)
paths.link_polychirp(COLLECTION)
K = 10                         # the 10 target species; non_target is the all-zeros negative


def specs() -> dict:
    from building.models import TinyChirpModel, mel_cnn_new, wrennet
    return {"mel_polychirp_x2": TinyChirpModel(base_model=mel_cnn_new.ULTRALIGHT, multi=2.0),
            "mel_polychirp_x3": TinyChirpModel(base_model=mel_cnn_new.ULTRALIGHT, multi=3.0),
            "wrennet": TinyChirpModel(base_model=wrennet.MODEL)}


def species(catalog, seed: int) -> tuple[list[int], int]:
    """(the target classes' indices, the non_target index), drawn as PolyChirp's run_scale does for
    run `seed` (k = 10 of the 10 species: only their order depends on the seed)."""
    pool = [i for i, n in enumerate(catalog.class_names) if n != "non_target"]
    return np.random.default_rng([seed, K, 0]).choice(pool, K, replace=False).tolist(), catalog.class_names.index("non_target")


# --- classification: PolyChirp's own training, with the paper's protocol ---------------------------

def install_protocol(runner) -> None:
    """Replaces, in PolyChirp's runner.train_cell, the compared validation loss by minus the
    validation macro AP (source checked, so that a change upstream fails loudly)."""
    from sklearn.metrics import average_precision_score

    def val_macro_ap(batches):
        y = np.concatenate([b[0] for b in batches])
        return float(average_precision_score(y, np.concatenate([b[1] for b in batches]), average="macro"))
    src = inspect.getsource(runner.train_cell)
    for old, new in (
        ("            val_losses: dict[int, list[float]] = {id(s): [] for s in active}\n",
         "            val_losses: dict[int, list[float]] = {id(s): [] for s in active}\n            val_preds: dict = {}\n"),
        ("                    out = s.model.test_on_batch(feats[s.spec.input_frontend], y, sample_weight=sw)\n",
         "                    out = s.model.test_on_batch(feats[s.spec.input_frontend], y, sample_weight=sw)\n"
         "                    val_preds.setdefault(id(s), []).append((np.asarray(y), np.asarray(s.model.predict_on_batch(feats[s.spec.input_frontend]))))\n"),
        ("                vloss = float(np.mean(val_losses[id(s)])) if val_losses[id(s)] else math.inf\n",
         "                vloss = -val_macro_ap(val_preds[id(s)])  # minimized\n"),
    ):
        assert src.count(old) == 1, f"PolyChirp's train_cell changed: {old.strip()!r}"
        src = src.replace(old, new)
    namespace = dict(vars(runner), val_macro_ap=val_macro_ap)
    exec(compile(src, inspect.getsourcefile(runner.train_cell), "exec"), namespace)
    runner.train_cell = namespace["train_cell"]


def train_classification(name: str, seed: int, run_dir: Path, protocol: dict) -> None:
    """One run of PolyChirp's scale benchmark (k = 10, one cell): PolyChirp trains, tunes the per-class
    thresholds, quantizes to INT8 TFLite and writes results.npz (every clip's float and INT8 scores on
    the tuning set, train + val, and on the test set) into run_dir."""
    import keras
    from building.polychirp_training import runner
    install_protocol(runner)
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
    return spec.builder(1), lambda audio: featurize(audio, spec.input_frontend)
