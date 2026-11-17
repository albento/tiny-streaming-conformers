"""From a trained ViT to the AudioMoth: INT8 quantization, C export, host inference, firmware build.

    uv run export.py runs/classification/vit_s24_l2/seed_0      # .bin to flash, in that run folder

Quantization (symmetric, zero-point 0, CMSIS-NN / TFLite conventions):
- int8 per-tensor weights; activation scales from percentiles of the activations of calibration
  spectrograms (99.9 for the PCEN input, 99.99 elsewhere), read by hooks on the model's own modules;
- int16 instead of int8 (same range, 256 finer steps) for the input, the stem, the patch embedding,
  the first convolution module and layer 0's Q, V and attention output: on quiet frames these vary by
  a fraction of an int8 step, which LayerNorm then amplifies (one PolyChirp run: macro-F2 0.94 float,
  0.81 all-int8). Everything after layer 0's first LayerNorm is int8;
- LayerNorm and its residual add in float; the attention scores requantized as their gap to the row
  max on a fixed 0.1-nat grid, softmax by a 129-entry exp table, ALiBi folded into the Q.K^T accumulator.
The firmware (firmware/vit_inference.c) reads vit_model.c/.h, the front end (firmware/frontend.c) pcen.c/.h.
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
def activations(vit, spec: torch.Tensor, batch: int = 64) -> dict[str, np.ndarray]:
    """|values| of every quantized tensor over calibration spectrograms [N, 129, T]: x (input),
    stem, e0 (patch embedding), per convolution module l c{l}.d / .p / .n (ReLU(depthwise),
    pointwise, LayerNorm), per layer l q, k, v, ao (attention output), o (out_proj), x1 (LayerNorm1),
    h (ReLU(linear1)), f (linear2), e (LayerNorm2)."""
    vit.eval()
    seen: dict[str, list] = {}
    keep = lambda name, t: seen.setdefault(name, []).append(t.abs().float().flatten().cpu().numpy())  # noqa: E731
    hooks = [vit.stem.register_forward_hook(lambda m, i, o: keep("stem", o)),
             vit.patch_embed.register_forward_hook(lambda m, i, o: keep("e0", o))]
    for l, cm in enumerate(vit.convs):
        hooks += [cm.depthwise.register_forward_hook(lambda m, i, o, l=l: keep(f"c{l}.d", torch.relu(o))),
                  cm.pointwise.register_forward_hook(lambda m, i, o, l=l: keep(f"c{l}.p", o)),
                  cm.norm.register_forward_hook(lambda m, i, o, l=l: keep(f"c{l}.n", o))]
    for l, layer in enumerate(vit.layers):
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
    device = vit.patch_embed.weight.device
    try:
        for start in range(0, spec.shape[0], batch):
            x = spec[start:start + batch].to(device)
            keep("x", x)
            vit.token_logits(x)
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

def write_pcen(vit, initial_m: np.ndarray, out: Path) -> None:
    """pcen.h/.c: the learned PCEN, one value per STFT bin. delta^r is exported as
    stftFastPow(delta, r), the firmware's own first term, so that a silent bin gives exactly 0."""
    v = {k: x.detach().cpu().double().numpy() for k, x in vit.spectrogram.pcen.values().items()}
    arrays = {"model_pcen_alpha": v["alpha"], "model_pcen_delta": v["delta"], "model_pcen_r": v["r"],
              "model_pcen_ema_alpha": v["smooth"], "model_pcen_delta_pow_r": fast_pow(v["delta"], v["r"]),
              "model_pcen_initial_m": initial_m}
    n = v["alpha"].size
    header = ["#ifndef MODEL_PCEN_H", "#define MODEL_PCEN_H", "", f"#define MODEL_PCEN_NUMBER_OF_BINS {n}",
              "#define MODEL_PCEN_EPS 1.0000000000e-06f", ""] + [f"extern const float {k}[{n}];" for k in arrays] + ["", "#endif"]
    (out / "pcen.h").write_text("\n".join(header) + "\n")
    (out / "pcen.c").write_text('#include "pcen.h"\n\n' + "\n".join(c_array("float", k, a, lambda x: f"{float(x):.10e}f")
                                                                     for k, a in arrays.items()) + "\n")


def write_c(vit, calib_spec: torch.Tensor, initial_m: torch.Tensor, out: Path, tag: str) -> None:
    """vit_model.h/.c and pcen.h/.c of a causal ViT, calibrated on PCEN spectrograms [N, 129, T]."""
    assert vit.causal, "only the causal ViT runs on the device"
    out.mkdir(parents=True, exist_ok=True)
    s = {k: float(np.percentile(a, 99.9 if k == "x" else 99.99)) / 127 for k, a in activations(vit, calib_spec).items()}
    d, heads, ctx, depth = vit.patch_embed.out_features, vit.alibi_slopes.shape[0], vit.context, len(vit.layers)
    conv, stem = vit.stem.conv, vit.stem
    header = ["#ifndef VIT_MODEL_H", "#define VIT_MODEL_H", "", "#include <stdint.h>", "",
              f"#define VIT_N_BINS {vit.spectrogram.pcen.raw_alpha.numel()}", f"#define VIT_PATCH_INPUTS {stem.out_features}",
              f"#define VIT_EMBED_DIM {d}", f"#define VIT_N_HEADS {heads}", f"#define VIT_HEAD_DIM {d // heads}",
              f"#define VIT_CONTEXT_SIZE {ctx}", f"#define VIT_FFN_DIM {vit.layers[0].linear1.out_features}",
              f"#define VIT_DEPTH {depth}", f"#define VIT_N_CLASSES {vit.n_classes}",
              f"#define VIT_INFERENCE_INPUT_FRAMES {vit.frames_per_buffer}", f'#define VIT_MODEL_TAG "{tag}"', "",
              "/* accScale: one int32 accumulator unit in float (projections feeding a LayerNorm residual in float) */",
              "typedef struct { const int8_t *weight; const int32_t *bias; int32_t multiplier; int32_t shift; float accScale; } vit_fc_t;",
              "typedef struct { float inputScaleA; float inputScaleB; const float *gamma; const float *beta; float outputInvScale; } vit_ln_t;",
              "typedef struct {", "    vit_fc_t q, k, v, outProj, ffn1, ffn2;", "    int32_t qkMultiplier, qkShift;",
              "    const int32_t *alibiAcc;  /* [VIT_N_HEADS][2 * VIT_CONTEXT_SIZE] */",
              "    float qkScale;            /* layer 0: one Q.K^T accumulator unit in nats */",
              "    const float *alibiSlope;  /* [VIT_N_HEADS], nats per token of age (layer 0) */",
              "    int32_t softmaxMultiplier, softmaxShift, softmaxDiffMin;", "    int32_t avMultiplier, avShift;",
              "    vit_ln_t ln1, ln2;", "} vit_layer_t;", "",
              "extern const vit_fc_t vit_patch_embed;", "extern const vit_layer_t vit_layers[VIT_DEPTH];",
              "/* stem: ReLU(Conv2d) over the last VIT_STEM_TIME_KERNEL frames, weight [channel][freq][time] */",
              "#define VIT_HAS_STEM 1", f"#define VIT_STEM_CHANNELS {conv.out_channels}",
              f"#define VIT_STEM_FREQ_KERNEL {conv.kernel_size[0]}", f"#define VIT_STEM_FREQ_STRIDE {conv.stride[0]}",
              f"#define VIT_STEM_FREQ_PAD {conv.padding[0]}", f"#define VIT_STEM_TIME_KERNEL {conv.kernel_size[1]}",
              f"#define VIT_STEM_TIME_STRIDE {conv.stride[1]}", f"#define VIT_STEM_OUT_BINS {stem.out_bins}",
              f"#define VIT_STEM_BUFFER_ALIGNED {int(stem.aligned)}", "extern const vit_fc_t vit_stem;",
              "/* convolution module before every layer: depthwise [channel][tap] (oldest first), pointwise,",
              "   residual + LayerNorm in float; module 0 in int16 */",
              f"#define VIT_CONV_KERNEL {vit.convs[0].kernel_size}", "#define VIT_CONV0_INT16 1",
              f"#define VIT_CONV0_IN_INV_SCALE {ffmt(1 / s['e0'])}",
              "typedef struct { vit_fc_t depthwise, pointwise; vit_ln_t ln; } vit_conv_t;",
              "extern const vit_conv_t vit_convs[VIT_DEPTH];",
              "extern const int8_t vit_classifier_weight[VIT_N_CLASSES * VIT_EMBED_DIM];",
              "extern const float vit_classifier_bias[VIT_N_CLASSES];", f"#define VIT_INT16_STEPS {INT16_STEPS}",
              f"#define VIT_INPUT_INV_SCALE {ffmt(INT16_STEPS / s['x'])}", f"#define VIT_EMBED_INV_SCALE {ffmt(INT16_STEPS / s['e0'])}",
              f"#define VIT_SCORE_DIFF_INV_SCALE {ffmt(1 / SCORE_DIFF_SCALE)}"]
    source = ['#include "vit_model.h"', ""]

    def fc(name, weight, bias, s_in, s_out):
        wq, ws = weight_quant(weight)
        mult, shift = quantize_multiplier(s_in * ws / s_out)
        source.extend(["static " + c_array("int8_t", f"{name}_w", wq), "static " + c_array("int32_t", f"{name}_b", bias_quant(bias, s_in * ws))])
        return f"{{ {name}_w, {name}_b, {mult}, {shift}, {ffmt(s_in * ws)} }}"

    def ln(name, norm, s_a, s_b, s_out):
        source.extend("static " + c_array("float", f"{name}_{k}", p.detach().cpu().numpy(), ffmt) for k, p in (("g", norm.weight), ("b", norm.bias)))
        return f"{{ {ffmt(s_a)}, {ffmt(s_b)}, {name}_g, {name}_b, {ffmt(1 / s_out)} }}"

    n16 = INT16_STEPS
    source.append(f"const vit_fc_t vit_stem = {fc('vit_stem', conv.weight.reshape(conv.out_channels, -1), conv.bias, s['x'] / n16, s['stem'] / n16)};")
    patch = fc("vit_pe", vit.patch_embed.weight, vit.patch_embed.bias, s["stem"] / n16, s["e0"] / n16)
    convs, layers = [], []
    slopes = vit.alibi_slopes.cpu().double().numpy()
    for l, (cm, layer) in enumerate(zip(vit.convs, vit.layers)):
        n = n16 if l == 0 else 1
        s_cin = s["e0"] if l == 0 else s[f"l{l - 1}.e"]
        dw = fc(f"vit_c{l}_dw", cm.depthwise.weight.reshape(d, -1), cm.depthwise.bias, s_cin / n, s[f"c{l}.d"] / n)
        pw = fc(f"vit_c{l}_pw", cm.pointwise.weight, cm.pointwise.bias, s[f"c{l}.d"] / n, s[f"c{l}.p"])
        convs.append(f"    {{ {dw}, {pw}, {ln(f'vit_c{l}_ln', cm.norm, s_cin, s[f'c{l}.p'], s[f'c{l}.n'] / n)} }}")
        sa, s_in = layer.self_attn, s[f"c{l}.n"]
        (wq, wk, wv), (bq, bk, bv) = sa.in_proj_weight.chunk(3), sa.in_proj_bias.chunk(3)
        q = fc(f"vit_l{l}_q", wq, bq, s_in / n, s[f"l{l}.q"] / n)
        k = fc(f"vit_l{l}_k", wk, bk, s_in / n, s[f"l{l}.k"])  # K stays int8
        v = fc(f"vit_l{l}_v", wv, bv, s_in / n, s[f"l{l}.v"] / n)
        hd = math.sqrt(d // heads)
        qk_mult, qk_shift = quantize_multiplier(s[f"l{l}.q"] * s[f"l{l}.k"] / (hd * SCORE_DIFF_SCALE))
        alibi = np.round(-slopes[:, None] * np.arange(ctx) * hd / (s[f"l{l}.q"] * s[f"l{l}.k"])).clip(-2**31, 2**31 - 1)
        source += ["static " + c_array("float", f"vit_l{l}_alibi_slope", slopes, ffmt),
                   "static " + c_array("int32_t", f"vit_l{l}_alibi", np.tile(alibi[:, ::-1], 2).astype(np.int32))]  # by age reversed, twice
        sm = softmax_params(SCORE_DIFF_SCALE)
        av_mult, av_shift = quantize_multiplier(s[f"l{l}.v"] / 256 / s[f"l{l}.ao"])
        o = fc(f"vit_l{l}_o", sa.out_proj.weight, sa.out_proj.bias, s[f"l{l}.ao"] / n, s[f"l{l}.o"])
        ln1 = ln(f"vit_l{l}_ln1", layer.norm1, s_in, s[f"l{l}.o"], s[f"l{l}.x1"])
        f1 = fc(f"vit_l{l}_f1", layer.linear1.weight, layer.linear1.bias, s[f"l{l}.x1"], s[f"l{l}.h"])
        f2 = fc(f"vit_l{l}_f2", layer.linear2.weight, layer.linear2.bias, s[f"l{l}.h"], s[f"l{l}.f"])
        ln2 = ln(f"vit_l{l}_ln2", layer.norm2, s[f"l{l}.x1"], s[f"l{l}.f"], s[f"l{l}.e"])
        qk_scale = (s[f"l{l}.q"] / n) * s[f"l{l}.k"] / hd
        layers.append(f"    {{ {q}, {k}, {v}, {o}, {f1}, {f2},\n      {qk_mult}, {qk_shift}, vit_l{l}_alibi, {ffmt(qk_scale)}, "
                      f"vit_l{l}_alibi_slope,\n      {sm[0]}, {sm[1]}, {sm[2]},\n      {av_mult}, {av_shift}, {ln1}, {ln2} }}")
    wc, wcs = weight_quant(vit.classifier.weight)
    header += [f"#define VIT_CLASSIFIER_LOGIT_SCALE {ffmt(s[f'l{depth - 1}.e'] * wcs)}", "#define VIT_SOFTMAX_TABLE_SIZE 129",
               "extern const uint16_t vit_softmax_exp_q15[VIT_SOFTMAX_TABLE_SIZE];", "", "#endif"]
    source += [c_array("uint16_t", "vit_softmax_exp_q15", np.round(np.exp(-SCORE_DIFF_SCALE * np.arange(129)) * 32768).astype(np.int64)),
               c_array("int8_t", "vit_classifier_weight", wc),
               c_array("float", "vit_classifier_bias", vit.classifier.bias.detach().cpu().numpy(), ffmt),
               f"const vit_fc_t vit_patch_embed = {patch};",
               "const vit_layer_t vit_layers[VIT_DEPTH] = {\n" + ",\n".join(layers) + "\n};",
               "const vit_conv_t vit_convs[VIT_DEPTH] = {\n" + ",\n".join(convs) + "\n};"]
    (out / "vit_model.h").write_text("\n".join(header) + "\n")
    (out / "vit_model.c").write_text("\n".join(source) + "\n")
    write_pcen(vit, initial_m.cpu().double().numpy(), out)


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
    sources = [FIRMWARE / f for f in ("host_infer.c", "frontend.c", "vit_inference.c")] + \
              [export_dir / "vit_model.c", export_dir / "pcen.c"] + [FIRMWARE / f for f in CMSIS_SOURCES]
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

def build_firmware(export_dir: Path, out_bin: Path, heap: str = "0x800") -> None:
    """The latency-measuring AudioMoth firmware for one export: AudioMoth-Project (build system) and
    AudioMoth-Firmware-Basic (firmware) as their submodules pin them, our C and the export copied in,
    firmware/main.patch applied, then make (arm-none-eabi-gcc)."""
    with tempfile.TemporaryDirectory() as tmp:
        b = Path(tmp) / "build"
        shutil.copytree(FIRMWARE / "AudioMoth-Project", b, ignore=shutil.ignore_patterns(".git"))
        for d in ("src", "inc"):
            shutil.copytree(FIRMWARE / "AudioMoth-Firmware-Basic" / d, b / d, dirs_exist_ok=True)
        for f in ("vit_inference.c", "frontend.c", "cmsis_dsp.c"):
            shutil.copy(FIRMWARE / f, b / "src")
        for f in ("vit_inference.h", "frontend.h", "stft_window.h"):
            shutil.copy(FIRMWARE / f, b / "inc")
        (b / "model").mkdir(exist_ok=True)
        for f in ("vit_model.c", "vit_model.h", "pcen.c", "pcen.h"):
            shutil.copy(export_dir / f, b / "model")
        for d in ("CMSIS-DSP", "CMSIS-NN", "CMSIS_6"):
            shutil.copytree(FIRMWARE / d, b / d, ignore=shutil.ignore_patterns(".git"), dirs_exist_ok=True)
        subprocess.run(["patch", "-s", "-p1", "-i", str(FIRMWARE / "main.patch")], cwd=b, check=True)
        (b / "build" / "objects").mkdir(parents=True, exist_ok=True)
        subprocess.run(["make", "-s", f"DFLAGS=-MMD -D__HEAP_SIZE={heap}"], cwd=b / "build", check=True)
        shutil.copy(b / "build" / "audiomoth.bin", out_bin)


def main():
    parser = argparse.ArgumentParser(description="Build the AudioMoth firmware (.bin) of a trained causal ViT run.")
    parser.add_argument("run", type=Path, help="a run folder of train.py (runs/<task>/<model>/seed_<n>)")
    args = parser.parse_args()
    export_dir = args.run / "export"
    if not (export_dir / "vit_model.c").exists():
        sys.exit(f"{export_dir}: no INT8 export (only causal ViT runs have one)")
    out = args.run / f"audiomoth_{args.run.parent.name}_{args.run.name}.bin"
    build_firmware(export_dir, out)
    print(out)


if __name__ == "__main__":
    main()
