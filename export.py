"""From a trained Conformer to the AudioMoth: INT8 quantization, C export, host inference, firmware build.

    uv run export.py runs/classification/conf_s24_l2/seed_0      # .bin to flash, in that run folder
    uv run export.py --all                                      # every causal Conformer (detection_stream, seed 0), into runs/audiomoth/

Quantization (symmetric, zero-point 0, CMSIS-NN / TFLite conventions):
- int8 per-tensor weights; activation scales from percentiles of the activations of calibration
  spectrograms (99.9 for the PCEN input, 99.99 elsewhere), read by hooks on the model's own modules;
- int16 instead of int8 (same range, 256 finer steps) for the input, the stem, the patch embedding,
  the first convolution module and layer 0's Q, V and attention output: on quiet frames these vary by
  a fraction of an int8 step, which LayerNorm then amplifies (one PolyChirp run: macro-F2 0.94 float,
  0.81 all-int8). Everything after layer 0's first LayerNorm is int8;
- LayerNorm and its residual add in float; the attention scores requantized as their gap to the row
  max on a fixed 0.1-nat grid, softmax by a 129-entry exp table, ALiBi folded into the Q.K^T accumulator.
The firmware (firmware/conformer_inference.c) reads conformer_model.c/.h, the front end (firmware/frontend.c) pcen.c/.h.
"""
import argparse
import copy
import math
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

FIRMWARE = Path(__file__).resolve().parent / "firmware"
SCORE_DIFF_SCALE = 0.1  # nats per int8 step of (score - row max)
INT16_STEPS = 256       # int16 steps per int8 step


# --- calibration ----------------------------------------------------------------------------------

@torch.no_grad()
def activations(conformer, spec: torch.Tensor, batch: int = 64) -> dict[str, np.ndarray]:
    """|values| of every quantized tensor over calibration spectrograms [N, 129, T]: x (input),
    stem, e0 (patch embedding), per convolution module l c{l}.d / .p / .n (ReLU(depthwise),
    pointwise, LayerNorm), per layer l q, k, v, ao (attention output), o (out_proj), x1 (LayerNorm1),
    h (ReLU(linear1)), f (linear2), e (LayerNorm2)."""
    conformer.eval()
    seen: dict[str, list] = {}
    keep = lambda name, t: seen.setdefault(name, []).append(t.abs().float().flatten().cpu().numpy())  # noqa: E731
    hooks = [conformer.conv_stem.register_forward_hook(lambda m, i, o: keep("stem", o)),
             conformer.patch_embed.register_forward_hook(lambda m, i, o: keep("e0", o))]
    for l, cm in enumerate(conformer.convs):
        hooks += [cm.depthwise.register_forward_hook(lambda m, i, o, l=l: keep(f"c{l}.d", torch.relu(o))),
                  cm.pointwise.register_forward_hook(lambda m, i, o, l=l: keep(f"c{l}.p", o)),
                  cm.norm.register_forward_hook(lambda m, i, o, l=l: keep(f"c{l}.n", o))]
    for l, layer in enumerate(conformer.layers):
        # nn.MultiheadAttention applies q/k/v and out_proj as functions: q/k/v rebuilt from in_proj,
        # out_proj's input from a copy whose out_proj is the identity
        plain = copy.deepcopy(layer.self_attn)
        torch.nn.init.eye_(plain.out_proj.weight)
        torch.nn.init.zeros_(plain.out_proj.bias)

        def attn(m, args, kwargs, out, l=l, plain=plain):
            q, k, v = F.linear(args[0], m.in_proj_weight, m.in_proj_bias).chunk(3, dim=-1)
            keep(f"l{l}.q", q), keep(f"l{l}.k", k), keep(f"l{l}.v", v)
            keep(f"l{l}.ao", plain(*args, **kwargs)[0]), keep(f"l{l}.o", out[0])
        hooks += [layer.self_attn.register_forward_hook(attn, with_kwargs=True),
                  layer.norm1.register_forward_hook(lambda m, i, o, l=l: keep(f"l{l}.x1", o)),
                  layer.linear1.register_forward_hook(lambda m, i, o, l=l: keep(f"l{l}.h", torch.relu(o))),
                  layer.linear2.register_forward_hook(lambda m, i, o, l=l: keep(f"l{l}.f", o)),
                  layer.norm2.register_forward_hook(lambda m, i, o, l=l: keep(f"l{l}.e", o))]
    device = conformer.patch_embed.weight.device
    try:
        for start in range(0, spec.shape[0], batch):
            x = spec[start:start + batch].to(device)
            keep("x", x)
            conformer.token_logits(x)
    finally:
        for h in hooks:
            h.remove()
    return {k: np.concatenate(v) for k, v in seen.items()}


# --- quantization helpers -------------------------------------------------------------------------

def quantize_multiplier(real: float) -> tuple[int, int]:
    """real = multiplier * 2^(shift - 31), multiplier in [2^30, 2^31) (TFLite QuantizeMultiplier)."""
    if real <= 0:
        return 0, 0
    mantissa, exponent = math.frexp(real)
    q = int(round(mantissa * (1 << 31)))
    if q == 1 << 31:
        q, exponent = q // 2, exponent + 1
    return (0, 0) if exponent < -31 else (q, exponent)


def softmax_params(input_scale: float, integer_bits: int = 5) -> tuple[int, int, int]:
    """(multiplier, shift, diff_min) of CMSIS-NN's arm_softmax_s8 (TFLite PreprocessSoftmaxScaling)."""
    mult, shift = quantize_multiplier(min(input_scale * (1 << (31 - integer_bits)), float((1 << 31) - 1)))
    assert shift >= 0
    return mult, shift, -int(math.floor(((1 << integer_bits) - 1) * (1 << (31 - integer_bits)) / (1 << shift)))


def weight_quant(w: torch.Tensor) -> tuple[np.ndarray, float]:
    w = w.detach().cpu().double().numpy()
    scale = float(np.abs(w).max()) / 127
    return np.clip(np.round(w / scale), -127, 127).astype(np.int8), scale


def bias_quant(b: torch.Tensor, scale: float) -> np.ndarray:
    return np.round(b.detach().cpu().double().numpy() / scale).clip(-2**31, 2**31 - 1).astype(np.int32)


def c_array(ctype: str, name: str, values, fmt=str) -> str:
    flat = np.asarray(values).flatten()
    return f"const {ctype} {name}[{flat.size}] = {{{', '.join(fmt(v) for v in flat)}}};"


def ffmt(v) -> str:
    return f"{float(v):.9e}f"


def fast_pow(x, a) -> np.ndarray:
    """The firmware's stftFastPow, bit for bit (float32, truncating casts)."""
    log2 = (np.asarray(x, np.float32).view(np.int32) >> 7) - np.int32(8320252)
    scaled = (log2.astype(np.float32) * np.asarray(a, np.float32)).astype(np.int32)
    return ((scaled + np.int32(8320252)) << 7).astype(np.int32).view(np.float32)


# --- C export --------------------------------------------------------------------------------------

def write_pcen(conformer, initial_m: np.ndarray, out: Path) -> None:
    """pcen.h/.c: the learned PCEN, one value per STFT bin. delta^r is exported as
    stftFastPow(delta, r), the firmware's own first term, so that a silent bin gives exactly 0."""
    v = {k: x.detach().cpu().double().numpy() for k, x in conformer.spectrogram.pcen.values().items()}
    arrays = {"model_pcen_alpha": v["alpha"], "model_pcen_delta": v["delta"], "model_pcen_r": v["r"],
              "model_pcen_ema_alpha": v["smooth"], "model_pcen_delta_pow_r": fast_pow(v["delta"], v["r"]),
              "model_pcen_initial_m": initial_m}
    n = v["alpha"].size
    header = ["#ifndef MODEL_PCEN_H", "#define MODEL_PCEN_H", "", f"#define MODEL_PCEN_NUMBER_OF_BINS {n}",
              "#define MODEL_PCEN_EPS 1.0000000000e-06f", ""] + [f"extern const float {k}[{n}];" for k in arrays] + ["", "#endif"]
    (out / "pcen.h").write_text("\n".join(header) + "\n")
    (out / "pcen.c").write_text('#include "pcen.h"\n\n' + "\n".join(c_array("float", k, a, lambda x: f"{float(x):.10e}f")
                                                                     for k, a in arrays.items()) + "\n")


def write_c(conformer, calib_spec: torch.Tensor, initial_m: torch.Tensor, out: Path, tag: str) -> None:
    """conformer_model.h/.c and pcen.h/.c of a causal Conformer, calibrated on PCEN spectrograms [N, 129, T]."""
    assert conformer.causal, "only the causal Conformer runs on the device"
    out.mkdir(parents=True, exist_ok=True)
    s = {k: float(np.percentile(a, 99.9 if k == "x" else 99.99)) / 127 for k, a in activations(conformer, calib_spec).items()}
    d, heads, ctx, depth = conformer.patch_embed.out_features, conformer.alibi_slopes.shape[0], conformer.context, len(conformer.layers)
    conv, stem = conformer.conv_stem.conv, conformer.conv_stem
    header = ["#ifndef CONF_MODEL_H", "#define CONF_MODEL_H", "", "#include <stdint.h>", "",
              f"#define CONF_N_BINS {conformer.spectrogram.pcen.raw_alpha.numel()}", f"#define CONF_PATCH_INPUTS {stem.out_features}",
              f"#define CONF_EMBED_DIM {d}", f"#define CONF_N_HEADS {heads}", f"#define CONF_HEAD_DIM {d // heads}",
              f"#define CONF_CONTEXT_SIZE {ctx}", f"#define CONF_FFN_DIM {conformer.layers[0].linear1.out_features}",
              f"#define CONF_DEPTH {depth}", f"#define CONF_N_CLASSES {conformer.n_classes}",
              f"#define CONF_INFERENCE_INPUT_FRAMES {conformer.frames_per_buffer}", f'#define CONF_MODEL_TAG "{tag}"', "",
              "/* accScale: one int32 accumulator unit in float (projections feeding a LayerNorm residual in float) */",
              "typedef struct { const int8_t *weight; const int32_t *bias; int32_t multiplier; int32_t shift; float accScale; } conf_fc_t;",
              "typedef struct { float inputScaleA; float inputScaleB; const float *gamma; const float *beta; float outputInvScale; } conf_ln_t;",
              "typedef struct {", "    conf_fc_t q, k, v, outProj, ffn1, ffn2;", "    int32_t qkMultiplier, qkShift;",
              "    const int32_t *alibiAcc;  /* [CONF_N_HEADS][2 * CONF_CONTEXT_SIZE] */",
              "    float qkScale;            /* layer 0: one Q.K^T accumulator unit in nats */",
              "    const float *alibiSlope;  /* [CONF_N_HEADS], nats per token of age (layer 0) */",
              "    int32_t softmaxMultiplier, softmaxShift, softmaxDiffMin;", "    int32_t avMultiplier, avShift;",
              "    conf_ln_t ln1, ln2;", "} conf_layer_t;", "",
              "extern const conf_fc_t conf_patch_embed;", "extern const conf_layer_t conf_layers[CONF_DEPTH];",
              "/* stem: ReLU(Conv2d) over the last CONF_STEM_TIME_KERNEL frames, weight [channel][freq][time] */",
              "#define CONF_HAS_STEM 1", f"#define CONF_STEM_CHANNELS {conv.out_channels}",
              f"#define CONF_STEM_FREQ_KERNEL {conv.kernel_size[0]}", f"#define CONF_STEM_FREQ_STRIDE {conv.stride[0]}",
              f"#define CONF_STEM_FREQ_PAD {conv.padding[0]}", f"#define CONF_STEM_TIME_KERNEL {conv.kernel_size[1]}",
              f"#define CONF_STEM_TIME_STRIDE {conv.stride[1]}", f"#define CONF_STEM_OUT_BINS {stem.out_bins}",
              f"#define CONF_STEM_BUFFER_ALIGNED {int(stem.aligned)}", "extern const conf_fc_t conf_stem;",
              "/* convolution module before every layer: depthwise [channel][tap] (oldest first), pointwise,",
              "   residual + LayerNorm in float; module 0 in int16 */",
              f"#define CONF_CONV_KERNEL {conformer.convs[0].kernel_size}", "#define CONF_CONV0_INT16 1",
              f"#define CONF_CONV0_IN_INV_SCALE {ffmt(1 / s['e0'])}",
              "typedef struct { conf_fc_t depthwise, pointwise; conf_ln_t ln; } conf_conv_t;",
              "extern const conf_conv_t conf_convs[CONF_DEPTH];",
              "extern const int8_t conf_classifier_weight[CONF_N_CLASSES * CONF_EMBED_DIM];",
              "extern const float conf_classifier_bias[CONF_N_CLASSES];", f"#define CONF_INT16_STEPS {INT16_STEPS}",
              f"#define CONF_INPUT_INV_SCALE {ffmt(INT16_STEPS / s['x'])}", f"#define CONF_EMBED_INV_SCALE {ffmt(INT16_STEPS / s['e0'])}",
              f"#define CONF_SCORE_DIFF_INV_SCALE {ffmt(1 / SCORE_DIFF_SCALE)}"]
    source = ['#include "conformer_model.h"', ""]

    def fc(name, weight, bias, s_in, s_out):
        wq, ws = weight_quant(weight)
        mult, shift = quantize_multiplier(s_in * ws / s_out)
        source.extend(["static " + c_array("int8_t", f"{name}_w", wq), "static " + c_array("int32_t", f"{name}_b", bias_quant(bias, s_in * ws))])
        return f"{{ {name}_w, {name}_b, {mult}, {shift}, {ffmt(s_in * ws)} }}"

    def ln(name, norm, s_a, s_b, s_out):
        source.extend("static " + c_array("float", f"{name}_{k}", p.detach().cpu().numpy(), ffmt) for k, p in (("g", norm.weight), ("b", norm.bias)))
        return f"{{ {ffmt(s_a)}, {ffmt(s_b)}, {name}_g, {name}_b, {ffmt(1 / s_out)} }}"

    n16 = INT16_STEPS
    source.append(f"const conf_fc_t conf_stem = {fc('conf_stem', conv.weight.reshape(conv.out_channels, -1), conv.bias, s['x'] / n16, s['stem'] / n16)};")
    patch = fc("conf_pe", conformer.patch_embed.weight, conformer.patch_embed.bias, s["stem"] / n16, s["e0"] / n16)
    convs, layers = [], []
    slopes = conformer.alibi_slopes.cpu().double().numpy()
    for l, (cm, layer) in enumerate(zip(conformer.convs, conformer.layers)):
        n = n16 if l == 0 else 1
        s_cin = s["e0"] if l == 0 else s[f"l{l - 1}.e"]
        dw = fc(f"conf_c{l}_dw", cm.depthwise.weight.reshape(d, -1), cm.depthwise.bias, s_cin / n, s[f"c{l}.d"] / n)
        pw = fc(f"conf_c{l}_pw", cm.pointwise.weight, cm.pointwise.bias, s[f"c{l}.d"] / n, s[f"c{l}.p"])
        convs.append(f"    {{ {dw}, {pw}, {ln(f'conf_c{l}_ln', cm.norm, s_cin, s[f'c{l}.p'], s[f'c{l}.n'] / n)} }}")
        sa, s_in = layer.self_attn, s[f"c{l}.n"]
        (wq, wk, wv), (bq, bk, bv) = sa.in_proj_weight.chunk(3), sa.in_proj_bias.chunk(3)
        q = fc(f"conf_l{l}_q", wq, bq, s_in / n, s[f"l{l}.q"] / n)
        k = fc(f"conf_l{l}_k", wk, bk, s_in / n, s[f"l{l}.k"])  # K stays int8
        v = fc(f"conf_l{l}_v", wv, bv, s_in / n, s[f"l{l}.v"] / n)
        hd = math.sqrt(d // heads)
        qk_mult, qk_shift = quantize_multiplier(s[f"l{l}.q"] * s[f"l{l}.k"] / (hd * SCORE_DIFF_SCALE))
        alibi = np.round(-slopes[:, None] * np.arange(ctx) * hd / (s[f"l{l}.q"] * s[f"l{l}.k"])).clip(-2**31, 2**31 - 1)
        source += ["static " + c_array("float", f"conf_l{l}_alibi_slope", slopes, ffmt),
                   "static " + c_array("int32_t", f"conf_l{l}_alibi", np.tile(alibi[:, ::-1], 2).astype(np.int32))]  # by age reversed, twice
        sm = softmax_params(SCORE_DIFF_SCALE)
        av_mult, av_shift = quantize_multiplier(s[f"l{l}.v"] / 256 / s[f"l{l}.ao"])
        o = fc(f"conf_l{l}_o", sa.out_proj.weight, sa.out_proj.bias, s[f"l{l}.ao"] / n, s[f"l{l}.o"])
        ln1 = ln(f"conf_l{l}_ln1", layer.norm1, s_in, s[f"l{l}.o"], s[f"l{l}.x1"])
        f1 = fc(f"conf_l{l}_f1", layer.linear1.weight, layer.linear1.bias, s[f"l{l}.x1"], s[f"l{l}.h"])
        f2 = fc(f"conf_l{l}_f2", layer.linear2.weight, layer.linear2.bias, s[f"l{l}.h"], s[f"l{l}.f"])
        ln2 = ln(f"conf_l{l}_ln2", layer.norm2, s[f"l{l}.x1"], s[f"l{l}.f"], s[f"l{l}.e"])
        qk_scale = (s[f"l{l}.q"] / n) * s[f"l{l}.k"] / hd
        layers.append(f"    {{ {q}, {k}, {v}, {o}, {f1}, {f2},\n      {qk_mult}, {qk_shift}, conf_l{l}_alibi, {ffmt(qk_scale)}, "
                      f"conf_l{l}_alibi_slope,\n      {sm[0]}, {sm[1]}, {sm[2]},\n      {av_mult}, {av_shift}, {ln1}, {ln2} }}")
    wc, wcs = weight_quant(conformer.classifier.weight)
    header += [f"#define CONF_CLASSIFIER_LOGIT_SCALE {ffmt(s[f'l{depth - 1}.e'] * wcs)}", "#define CONF_SOFTMAX_TABLE_SIZE 129",
               "extern const uint16_t conf_softmax_exp_q15[CONF_SOFTMAX_TABLE_SIZE];", "", "#endif"]
    source += [c_array("uint16_t", "conf_softmax_exp_q15", np.round(np.exp(-SCORE_DIFF_SCALE * np.arange(129)) * 32768).astype(np.int64)),
               c_array("int8_t", "conf_classifier_weight", wc),
               c_array("float", "conf_classifier_bias", conformer.classifier.bias.detach().cpu().numpy(), ffmt),
               f"const conf_fc_t conf_patch_embed = {patch};",
               "const conf_layer_t conf_layers[CONF_DEPTH] = {\n" + ",\n".join(layers) + "\n};",
               "const conf_conv_t conf_convs[CONF_DEPTH] = {\n" + ",\n".join(convs) + "\n};"]
    (out / "conformer_model.h").write_text("\n".join(header) + "\n")
    (out / "conformer_model.c").write_text("\n".join(source) + "\n")
    write_pcen(conformer, initial_m.cpu().double().numpy(), out)


# --- host inference --------------------------------------------------------------------------------

CMSIS_SOURCES = ["CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_s8.c",
                 "CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s8.c",
                 "CMSIS-NN/Source/SoftmaxFunctions/arm_softmax_s8.c",
                 "CMSIS-NN/Source/SoftmaxFunctions/arm_nn_softmax_common_s8.c"] + [
    f"CMSIS-DSP/Source/{p}.c" for p in ("TransformFunctions/arm_rfft_fast_f32", "TransformFunctions/arm_rfft_fast_init_f32",
                                        "TransformFunctions/arm_cfft_f32", "TransformFunctions/arm_cfft_init_f32",
                                        "TransformFunctions/arm_cfft_radix8_f32", "TransformFunctions/arm_bitreversal2",
                                        "CommonTables/arm_common_tables", "CommonTables/arm_const_structs")]


def build_host(export_dir: Path) -> Path:
    """export_dir/host_infer: the firmware C compiled for the host, with CMSIS-NN's portable C
    kernels (bit-exact with their Cortex-M4 ones). -std=c99 keeps GCC from contracting
    floating-point operations, as in the firmware build."""
    binary = export_dir / "host_infer"
    includes = [FIRMWARE, export_dir, FIRMWARE / "CMSIS-NN/Include", FIRMWARE / "CMSIS-DSP/Include", FIRMWARE / "CMSIS-DSP/PrivateInclude"]
    sources = [FIRMWARE / f for f in ("host_infer.c", "frontend.c", "conformer_inference.c")] + \
              [export_dir / "conformer_model.c", export_dir / "pcen.c"] + [FIRMWARE / f for f in CMSIS_SOURCES]
    subprocess.run(["gcc", "-O2", "-std=c99", "-D__GNUC_PYTHON__", "-Wall", "-Wno-unused-function", "-Wno-unused-variable",
                    *[f"-I{p}" for p in includes], "-o", str(binary), *map(str, sources), "-lm"], check=True)
    return binary


def host_logits(binary: Path, audio: np.ndarray, first_frame_seed: bool = False, processes: int = 8) -> np.ndarray:
    """INT8 buffer logits [N, buffers, classes] of int16 recordings [N, samples] (samples a multiple
    of 1024), each streamed from its start as on the device, recordings split over processes."""
    shards = [s for s in np.array_split(np.arange(audio.shape[0]), processes) if s.size]

    def run(shard):
        with tempfile.TemporaryDirectory() as tmp:
            inp, out = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
            with open(inp, "wb") as fh:
                np.array([shard.size, audio.shape[1]], np.int32).tofile(fh)
                np.ascontiguousarray(audio[shard], dtype=np.int16).tofile(fh)
            subprocess.run([str(binary), str(inp), str(out), str(int(first_frame_seed)), "1"], check=True)
            raw = out.read_bytes()
            n, b, c = np.frombuffer(raw[:12], np.int32)
            return np.frombuffer(raw[12:], np.float32).reshape(n, b, c)
    with ThreadPoolExecutor(len(shards)) as pool:
        return np.concatenate(list(pool.map(run, shards)))


# --- AudioMoth firmware ----------------------------------------------------------------------------

class RamOverflow(RuntimeError):
    """The linker's "RAM memory overflowed!": static data + heap + stack over the 32 KB of the EFM32."""


def build_firmware(export_dir: Path, out_bin: Path, heap: str = "0x800", no_frequency_trigger: bool = False) -> None:
    """The latency-measuring AudioMoth firmware for one export: AudioMoth-Project (build system) and
    AudioMoth-Firmware-Basic (firmware) as their submodules pin them, our C and the export copied in,
    main.c, digitalfilter.c and the Makefile replaced by ours (firmware/audiomoth/), then make
    (arm-none-eabi-gcc). no_frequency_trigger: without the official firmware's frequency trigger, unused
    by this firmware, whose Hamming window takes 4 KB of RAM."""
    with tempfile.TemporaryDirectory() as tmp:
        b = Path(tmp) / "build"
        shutil.copytree(FIRMWARE / "AudioMoth-Project", b, ignore=shutil.ignore_patterns(".git"))
        for d in ("src", "inc"):
            shutil.copytree(FIRMWARE / "AudioMoth-Firmware-Basic" / d, b / d, dirs_exist_ok=True)
        for f in ("conformer_inference.c", "frontend.c", "cmsis_dsp.c"):
            shutil.copy(FIRMWARE / f, b / "src")
        for f in ("conformer_inference.h", "frontend.h", "stft_window.h"):
            shutil.copy(FIRMWARE / f, b / "inc")
        (b / "model").mkdir(exist_ok=True)
        for f in ("conformer_model.c", "conformer_model.h", "pcen.c", "pcen.h"):
            shutil.copy(export_dir / f, b / "model")
        for d in ("CMSIS-DSP", "CMSIS-NN", "CMSIS_6"):
            shutil.copytree(FIRMWARE / d, b / d, ignore=shutil.ignore_patterns(".git"), dirs_exist_ok=True)
        for f in ("main.c", "digitalfilter.c"):          # the official firmware's, with the Conformer added
            shutil.copy(FIRMWARE / "audiomoth" / f, b / "src")
        shutil.copy(FIRMWARE / "audiomoth" / "Makefile", b / "build")
        (b / "build" / "objects").mkdir(parents=True, exist_ok=True)
        flags = f"-MMD -D__HEAP_SIZE={heap}" + (" -DCONF_NO_FREQUENCY_TRIGGER" if no_frequency_trigger else "")
        r = subprocess.run(["make", "-s", f"DFLAGS={flags}"], cwd=b / "build", capture_output=True, text=True)
        if r.returncode:
            if "RAM memory overflowed" in r.stderr:
                raise RamOverflow(out_bin.name)
            sys.stderr.write(r.stderr)
            raise subprocess.CalledProcessError(r.returncode, r.args)
        shutil.copy(b / "build" / "audiomoth.bin", out_bin)


def build_fitting(export_dir: Path, out_bin: Path) -> Path:
    """build_firmware, again without the frequency trigger (4 KB of RAM) if the RAM overflows; the .bin
    is then named ..._no_freq_trigger.bin. Returns its path."""
    try:
        build_firmware(export_dir, out_bin)
        return out_bin
    except RamOverflow:
        out_bin = out_bin.with_name(out_bin.stem + "_no_freq_trigger.bin")
        print(f"{export_dir.parent.parent.name}: RAM overflowed, built again without the frequency trigger's Hamming window", flush=True)
        build_firmware(export_dir, out_bin, no_frequency_trigger=True)
        return out_bin


def causal_conformer_runs(experiment: str, seed: int) -> list[Path]:
    """The run of every causal Conformer of an experiment with that seed and an INT8 export."""
    from paths import RUNS
    return sorted(r for r in (RUNS / experiment).glob(f"conf_*/seed_{seed}")
                  if not r.parent.name.endswith("_nc") and (r / "export" / "conformer_model.c").exists())


def main():
    from paths import RUNS
    parser = argparse.ArgumentParser(description="Build the AudioMoth firmware (.bin) of a trained causal Conformer run, or of all of them.")
    parser.add_argument("run", type=Path, nargs="?", help="a run folder of train.py (runs/<task>/<model>/seed_<n>)")
    parser.add_argument("--all", action="store_true", help="every causal Conformer of --experiment, seed --seed, into --out")
    parser.add_argument("--experiment", default="detection_stream", help="with --all (default: detection_stream, one output)")
    parser.add_argument("--seed", type=int, default=0, help="with --all (default: 0; the latency does not depend on it)")
    parser.add_argument("--out", type=Path, default=RUNS / "audiomoth", help="with --all: the folder of the .bin files")
    args = parser.parse_args()
    if args.all == (args.run is not None):
        parser.error("give a run folder or --all")
    if args.all:
        runs = causal_conformer_runs(args.experiment, args.seed)
        if not runs:
            sys.exit(f"no causal Conformer run with an INT8 export in {RUNS / args.experiment}/*/seed_{args.seed}")
        args.out.mkdir(parents=True, exist_ok=True)
        failed = []
        for run in runs:
            out = args.out / f"audiomoth_{run.parent.name}.bin"
            print(f"{run.parent.name}: building", flush=True)
            try:
                print(build_fitting(run / "export", out), flush=True)
            except (subprocess.CalledProcessError, RamOverflow):
                failed.append(run.parent.name)
                print(f"{run.parent.name}: build FAILED (see the messages above)", flush=True)
        if failed:
            sys.exit(f"{len(runs) - len(failed)} built, failed: {', '.join(failed)}")
        return
    export_dir = args.run / "export"
    if not (export_dir / "conformer_model.c").exists():
        sys.exit(f"{export_dir}: no INT8 export (only causal Conformer runs have one)")
    print(build_fitting(export_dir, args.run / f"audiomoth_{args.run.parent.name}_{args.run.name}.bin"))


if __name__ == "__main__":
    main()
