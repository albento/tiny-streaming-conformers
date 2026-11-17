/****************************************************************************
 * vit_inference.c
 * openacousticdevices.info
 *****************************************************************************/

/* Streaming INT8 inference of the ViT of vit.py, one token per PCEN frame (per stem stride),
   matching its eval-mode forward: patch_embed -> VIT_DEPTH (>= 1) post-LN encoder layers
   (causal ALiBi attention over the last VIT_CONTEXT_SIZE tokens, ReLU FFN) -> classifier,
   token logits averaged per buffer. Weights/scales come from vit_model.c (export.py).

   Quantization: symmetric int8 weights and activations (zero-point 0), except softmax's output,
   which uses CMSIS-NN/TFLite's zero-point -128 (prob = (q + 128) / 256). Scores are requantized
   as their gap to the row max on a fixed grid (see export.py), not on an absolute scale.
   LayerNorm (with its residual add) runs in float.

   The input, the patch embedding and layer 0's Q, V (and its V history) and attention output are
   int16 instead of int8 - same range, VIT_INT16_STEPS finer steps (layer 0's K stays int8: no
   measurable difference, and 1 KB less RAM) - and
   LayerNorm1's two terms (the layer input and out_proj's output) are taken in float from the
   int32 accumulators of their projections. On quiet frames (PCEN ~ 0) these activations vary
   by a fraction of an int8 step, which LayerNorm1 then amplifies: in int8 they wiped out what
   the model reads on most frames of a bird recording (PolyChirp's Paris benchmark: one run
   macro-F2 0.94 float, 0.81 INT8). From LayerNorm1's output on, and in the next
   layers, everything is int8 as before. Weights stay int8 everywhere.

   Convolutional stem (VIT_HAS_STEM, vit.py's Stem): every frame's
   int16 input goes into a history of the last VIT_STEM_TIME_KERNEL frames (zeros before the first,
   the model's causal zero padding); every VIT_STEM_TIME_STRIDE frames (frame count since the
   reset, across buffers), ReLU(Conv2d) over that window gives the patch embedding's input and
   one token is run. Each frame adds the logits of the latest token (the one computed at or before
   it) to the buffer's mean - TinyViT's token logits repeated over their time_stride frames, then
   averaged per buffer (vit.py ViT.forward).

   Buffer-aligned stem (VIT_STEM_BUFFER_ALIGNED: time kernel = time stride = the buffer's 7
   frames): one token per call, whose stem window is the buffer's own frames, read straight from
   the spectrogram - no input history kept between buffers; the buffer's logits are that token's.

   Convolution module before every layer (VIT_CONV_KERNEL > 0, vit.py's ConvModule): a depthwise convolution over each channel's last VIT_CONV_KERNEL tokens (kept
   per module), ReLU, a pointwise projection, the residual add and a LayerNorm in float. Module 0
   sits between the patch embedding and layer 0 and hands layer 0 what the patch embedding would
   (int16 for Q/K/V, float for LayerNorm1's residual); inside, it is int16 like layer 0
   (VIT_CONV0_INT16) or int8. The next modules are int8, like their layers.

   Builds unchanged on a host for checking against the PyTorch model - CMSIS-NN's pure-C
   kernels are bit-exact with its Cortex-M4 DSP ones. */

#include <stdint.h>
#include <string.h>
#include <math.h>

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

#include "vit_inference.h"

#if VIT_HAS_STEM && !defined(VIT_STEM_BUFFER_ALIGNED)
#define VIT_STEM_BUFFER_ALIGNED                  0
#endif

#if VIT_HAS_STEM && VIT_STEM_BUFFER_ALIGNED && (VIT_STEM_TIME_KERNEL != VIT_INFERENCE_INPUT_FRAMES || VIT_STEM_TIME_STRIDE != VIT_INFERENCE_INPUT_FRAMES)
#error "a buffer-aligned stem must cover exactly one buffer's frames"
#endif

#define VIT_STEM_HISTORY                         (VIT_HAS_STEM && !VIT_STEM_BUFFER_ALIGNED)

#if VIT_DEPTH < 1
#error "vit_inference.c needs at least one layer"
#endif

/* layer 0 is int16 (runLayer0); the int8 layers (runLayer: K/V history vWindowT, attention
   runAttention) exist only with VIT_DEPTH >= 2 - with a single layer, its output goes straight to
   the classifier */

#define VIT_HAS_INT8_LAYERS                      (VIT_DEPTH > 1)

#if defined(__arm__)
#include "em_device.h"
#define VIT_GET_CYCLES()                         (DWT->CYCCNT)
#else
#define VIT_GET_CYCLES()                         0u
#endif

#define VIT_TIME(field, code) { \
    uint32_t vitStartCycles = VIT_GET_CYCLES(); \
    code; \
    timing->field += VIT_GET_CYCLES() - vitStartCycles; \
}

#define VIT_LAYER_NORM_EPS                       1.0e-5f

#ifndef VIT_CONV_KERNEL
#define VIT_CONV_KERNEL                          0
#endif

/* 1 = table softmax (runSoftmaxTable below), 0 = CMSIS-NN's arm_softmax_s8 - measured on the
   device at ~1100 cycles per element, over half of the whole ISR. Both take the same int8
   input and produce the same output convention. Layer 0 always uses the table. */

#ifndef VIT_SOFTMAX_LUT
#define VIT_SOFTMAX_LUT                          1
#endif

/* Per-layer, per-head history of the last VIT_CONTEXT_SIZE tokens' K and V, indexed by ring
   position. V is stored transposed ([head_dim][context]) so attn.V is a plain vec x mat^T.
   K in int8 for every layer; V in int16 for layer 0, in int8 for the next ones (index layer - 1).
   All layers advance together, so they share one ring position. Internal RAM. */

static int8_t kWindow[VIT_DEPTH][VIT_N_HEADS][VIT_CONTEXT_SIZE][VIT_HEAD_DIM];

static int16_t vWindowT16[VIT_N_HEADS][VIT_HEAD_DIM][VIT_CONTEXT_SIZE];

#if VIT_HAS_INT8_LAYERS
static int8_t vWindowT[VIT_DEPTH - 1][VIT_N_HEADS][VIT_HEAD_DIM][VIT_CONTEXT_SIZE];
#endif

#if VIT_CONV_KERNEL

/* each convolution module's last VIT_CONV_KERNEL input tokens, per channel and oldest first (the
   depthwise weight's layout); module 0 in convHistory16 if it is int16 */

#if VIT_CONV0_INT16
static int16_t convHistory16[VIT_EMBED_DIM][VIT_CONV_KERNEL];
#endif

#define VIT_CONV_INT8_MODULES                    (VIT_DEPTH - VIT_CONV0_INT16)

#if VIT_CONV_INT8_MODULES > 0
static int8_t convHistory[VIT_CONV_INT8_MODULES][VIT_EMBED_DIM][VIT_CONV_KERNEL];
#endif

#endif

static uint32_t currentPosition;

static uint32_t validCount;

#if VIT_STEM_HISTORY

/* the last VIT_STEM_TIME_KERNEL input frames (int16), per bin and oldest first - the stem
   weight's [freq][time] layout, so the window of out_bin p is one contiguous run of
   stemHistory - and the logits of the latest token, added to the buffer mean by every frame
   until the next token */

static int16_t stemHistory[VIT_N_BINS][VIT_STEM_TIME_KERNEL];

static uint32_t frameCount;

static float lastTokenLogits[VIT_N_CLASSES];

#endif

void ViTInference_reset(void) {

    memset(vWindowT16, 0, sizeof(vWindowT16));

    memset(kWindow, 0, sizeof(kWindow));

#if VIT_HAS_INT8_LAYERS
    memset(vWindowT, 0, sizeof(vWindowT));
#endif

    currentPosition = VIT_CONTEXT_SIZE - 1;

    validCount = 0;

#if VIT_CONV_KERNEL

#if VIT_CONV0_INT16
    memset(convHistory16, 0, sizeof(convHistory16));
#endif

#if VIT_CONV_INT8_MODULES > 0
    memset(convHistory, 0, sizeof(convHistory));
#endif

#endif

#if VIT_STEM_HISTORY

    memset(stemHistory, 0, sizeof(stemHistory));

    memset(lastTokenLogits, 0, sizeof(lastTokenLogits));

    frameCount = 0;

#endif

}

static inline int8_t clampToInt8(int32_t value) {

    return (int8_t)(value < -128 ? -128 : (value > 127 ? 127 : value));

}

static inline int16_t clampToInt16(int32_t value) {

    return (int16_t)(value < -32768 ? -32768 : (value > 32767 ? 32767 : value));

}

static inline int32_t roundToInt(float value) {

    return (int32_t)(value >= 0.0f ? value + 0.5f : value - 0.5f);

}

/* sum + weight[0..n) . input[0..n) in int32, int8 weights x int16 / int8 inputs. On the Cortex-M4
   (ARM_MATH_DSP), CMSIS-NN's own inner loop (arm_nn_vec_mat_mult_t_s16 / _s8): 4 int8 widened to two
   int16 pairs (read_and_pad, in order for the int16 input; read_and_pad_reordered on both sides for
   int8 x int8), two products per SMLAD. The same integers as the plain loop, which runs the remainder
   and the host build. */

static inline int32_t dotS8S16(const int8_t *weight, const int16_t *input, int32_t n, int32_t sum) {

    int32_t i = 0;

#if defined(ARM_MATH_DSP)

    for (; i + 4 <= n; i += 4) {

        int32_t w01, w23;

        weight = read_and_pad(weight, &w01, &w23);

        sum = SMLAD(w01, arm_nn_read_q15x2_ia(&input), sum);

        sum = SMLAD(w23, arm_nn_read_q15x2_ia(&input), sum);

    }

#endif

    for (; i < n; i += 1) sum += (int32_t)*weight++ * (int32_t)*input++;

    return sum;

}

static inline int32_t dotS8S8(const int8_t *a, const int8_t *b, int32_t n, int32_t sum) {

    int32_t i = 0;

#if defined(ARM_MATH_DSP)

    for (; i + 4 <= n; i += 4) {

        int32_t a02, a13, b02, b13;

        a = read_and_pad_reordered(a, &a02, &a13);

        b = read_and_pad_reordered(b, &b02, &b13);

        sum = SMLAD(a02, b02, sum);

        sum = SMLAD(a13, b13, sum);

    }

#endif

    for (; i < n; i += 1) sum += (int32_t)*a++ * (int32_t)*b++;

    return sum;

}

static void runFullyConnected(const vit_fc_t *fc, const int8_t *input, int32_t inputSize, int8_t *output, int32_t outputSize, int32_t activationMin) {

    cmsis_nn_context ctx = { NULL, 0 };

    cmsis_nn_fc_params fcParams = { 0, 0, 0, { activationMin, 127 } };

    cmsis_nn_per_tensor_quant_params quantParams = { fc->multiplier, fc->shift };

    /* arm_fully_connected_s8 reads filter_dims->n as the accumulation depth (input size) and
       output_dims->c as the number of outputs. */

    cmsis_nn_dims inputDims = { 1, 1, 1, inputSize };

    cmsis_nn_dims filterDims = { inputSize, 1, 1, outputSize };

    cmsis_nn_dims biasDims = { 1, 1, 1, outputSize };

    cmsis_nn_dims outputDims = { 1, 1, 1, outputSize };

    arm_fully_connected_s8(&ctx, &fcParams, &quantParams, &inputDims, input, &filterDims, fc->weight, &biasDims, fc->bias, &outputDims, output);

}

/* int32 accumulators of an int8-weight projection (bias included), for an int8 or an int16
   input. int16 x int8 products: at most 32768 * 127 per term, ~1.7e8 over the 40 inputs of the
   patch embedding - no overflow. */

#if VIT_HAS_INT8_LAYERS || (VIT_CONV_KERNEL && !VIT_CONV0_INT16)

static void accumulateInt8Input(const vit_fc_t *fc, const int8_t *input, int32_t inputSize, int32_t *acc, int32_t outputSize) {

    for (int32_t o = 0; o < outputSize; o += 1) acc[o] = dotS8S8(fc->weight + o * inputSize, input, inputSize, fc->bias[o]);

}

#endif

static void accumulateInt16Input(const vit_fc_t *fc, const int16_t *input, int32_t inputSize, int32_t *acc, int32_t outputSize) {

    for (int32_t o = 0; o < outputSize; o += 1) acc[o] = dotS8S16(fc->weight + o * inputSize, input, inputSize, fc->bias[o]);

}

/* int16 input -> int16 or int8 output (requantized with the projection's multiplier/shift) */

static void runFullyConnectedInt16(const vit_fc_t *fc, const int16_t *input, int32_t inputSize, int16_t *output, int32_t outputSize) {

    for (int32_t o = 0; o < outputSize; o += 1) {

        output[o] = clampToInt16(arm_nn_requantize(dotS8S16(fc->weight + o * inputSize, input, inputSize, fc->bias[o]), fc->multiplier, fc->shift));

    }

}

static void runFullyConnectedInt16ToInt8(const vit_fc_t *fc, const int16_t *input, int32_t inputSize, int8_t *output, int32_t outputSize) {

    for (int32_t o = 0; o < outputSize; o += 1) {

        output[o] = clampToInt8(arm_nn_requantize(dotS8S16(fc->weight + o * inputSize, input, inputSize, fc->bias[o]), fc->multiplier, fc->shift));

    }

}

/* LayerNorm of x, in place, in float */

static void layerNormFloat(const vit_ln_t *ln, float *x) {

    float sum = 0.0f;

    for (uint32_t i = 0; i < VIT_EMBED_DIM; i += 1) sum += x[i];

    float mean = sum / (float)VIT_EMBED_DIM;

    float sumSquaredDiff = 0.0f;

    for (uint32_t i = 0; i < VIT_EMBED_DIM; i += 1) {

        float diff = x[i] - mean;

        sumSquaredDiff += diff * diff;

    }

    float invStd = 1.0f / sqrtf(sumSquaredDiff / (float)VIT_EMBED_DIM + VIT_LAYER_NORM_EPS);

    for (uint32_t i = 0; i < VIT_EMBED_DIM; i += 1) x[i] = (x[i] - mean) * invStd * ln->gamma[i] + ln->beta[i];

}

/* LayerNorm of x, in place, requantized to int8 */

static void normaliseToInt8(const vit_ln_t *ln, float *x, int8_t *output) {

    layerNormFloat(ln, x);

    for (uint32_t i = 0; i < VIT_EMBED_DIM; i += 1) output[i] = clampToInt8(roundToInt(x[i] * ln->outputInvScale));

}

/* LayerNorm1: residual (float: residualFloat, else ln1.inputScaleA * residualInt8) + out_proj's
   int32 accumulators in float */

static void runLayerNorm1(const vit_ln_t *ln, const float *residualFloat, const int8_t *residualInt8, const int32_t *outProjAcc, float outProjAccScale, int8_t *output) {

    float x[VIT_EMBED_DIM];

    for (uint32_t i = 0; i < VIT_EMBED_DIM; i += 1) {

        float residual = residualFloat != NULL ? residualFloat[i] : ln->inputScaleA * (float)residualInt8[i];

        x[i] = residual + (float)outProjAcc[i] * outProjAccScale;

    }

    normaliseToInt8(ln, x, output);

}

/* LayerNorm2: scaleA * a + scaleB * b, a and b int8 */

static void runLayerNorm2(const vit_ln_t *ln, const int8_t *a, const int8_t *b, int8_t *output) {

    float x[VIT_EMBED_DIM];

    for (uint32_t i = 0; i < VIT_EMBED_DIM; i += 1) x[i] = ln->inputScaleA * (float)a[i] + ln->inputScaleB * (float)b[i];

    normaliseToInt8(ln, x, output);

}

/* Scores arrive as (score - row max) on a fixed grid, so every value is in [-128, 0] and exp()
   is a lookup in vit_softmax_exp_q15 (Q15, exp(0) = 32768). One integer reciprocal per row
   replaces arm_softmax_s8's two fixed-point exp() per element. Output convention unchanged:
   prob = (q + 128) / 256, i.e. q = round(256 * e / sum) - 128. The row max contributes
   exp(0), so sum >= 2^15, reciprocal <= 2^16 and e * reciprocal <= 2^31 - no overflow. */

static void runSoftmaxTable(const int8_t *scores, uint32_t count, int8_t *probs) {

    uint32_t sum = 0;

    for (uint32_t j = 0; j < count; j += 1) sum += vit_softmax_exp_q15[-scores[j]];

    uint32_t reciprocal = 0x80000000u / sum;

    for (uint32_t j = 0; j < count; j += 1) {

        uint32_t prob = ((uint32_t)vit_softmax_exp_q15[-scores[j]] * reciprocal + (1u << 22)) >> 23;

        probs[j] = (int8_t)((int32_t)(prob > 255 ? 255 : prob) - 128);

    }

}

/* Layer 0's attention: int16 Q, int8 K, int16 V. Q.K^T over one head (at most 8 * 32768 * 128
   ~ 3.4e7 in int32) is turned into nats in float (qkScale), with ALiBi added in nats
   (alibiSlope); the gap to the row max then goes on the same fixed grid as the int8 layers
   (VIT_SCORE_DIFF_INV_SCALE) and through the same table softmax. attn.V: probabilities (0..255)
   x int16 V, at most 32 * 255 * 32768 ~ 2.7e8 in int32, requantized to int16. */

static void runAttentionInt16(const int16_t *q, int16_t *attnOut, ViTInferenceTiming_t *timing) {

    const vit_layer_t *params = &vit_layers[0];

    for (uint32_t h = 0; h < VIT_N_HEADS; h += 1) {

        int8_t scores[VIT_CONTEXT_SIZE];

        int8_t probs[VIT_CONTEXT_SIZE];

        VIT_TIME(qkCycles[0], {

            const int16_t *qHead = q + h * VIT_HEAD_DIM;

            float score[VIT_CONTEXT_SIZE];

            float scoreMax = -INFINITY;

            for (uint32_t j = 0; j < validCount; j += 1) {

                int32_t acc = dotS8S16(kWindow[0][h][j], qHead, VIT_HEAD_DIM, 0);

                /* ring entry j's age: 0 for the current token */

                uint32_t age = (currentPosition + VIT_CONTEXT_SIZE - j) % VIT_CONTEXT_SIZE;

                score[j] = (float)acc * params->qkScale - params->alibiSlope[h] * (float)age;

                if (score[j] > scoreMax) scoreMax = score[j];

            }

            for (uint32_t j = 0; j < validCount; j += 1) {

                int32_t diff = roundToInt((score[j] - scoreMax) * VIT_SCORE_DIFF_INV_SCALE);

                scores[j] = (int8_t)(diff < -128 ? -128 : diff);

            }

        })

        VIT_TIME(softmaxCycles[0], {

            runSoftmaxTable(scores, validCount, probs);

        })

        VIT_TIME(avCycles[0], {

            for (uint32_t i = 0; i < VIT_HEAD_DIM; i += 1) {

                const int16_t *vColumn = vWindowT16[h][i];

                /* sum (probs + 128) * v = probs . v + 128 * sum v: exact in int32 */

                int32_t vSum = 0;

                for (uint32_t j = 0; j < validCount; j += 1) vSum += vColumn[j];

                int32_t acc = dotS8S16(probs, vColumn, (int32_t)validCount, 128 * vSum);

                attnOut[h * VIT_HEAD_DIM + i] = clampToInt16(arm_nn_requantize(acc, params->avMultiplier, params->avShift));

            }

        })

    }

}

#if VIT_HAS_INT8_LAYERS

/* The int8 layers' attention (layer >= 1, history index layer - 1) */

static void runAttention(uint32_t layer, const int8_t *q, int8_t *attnOut, ViTInferenceTiming_t *timing) {

    const vit_layer_t *params = &vit_layers[layer];

    for (uint32_t h = 0; h < VIT_N_HEADS; h += 1) {

        int8_t scores[VIT_CONTEXT_SIZE];

        int8_t probs[VIT_CONTEXT_SIZE];

        VIT_TIME(qkCycles[layer], {

            /* alibiAcc holds each head's ALiBi bias (already in QK accumulator units) by age,
               reversed and doubled, so that alibi[j] is ring entry j's bias for the current
               ring position without any per-token rotation. Before the ring has wrapped, the
               valid entries are exactly 0..validCount-1. */

            const int32_t *alibi = params->alibiAcc + h * 2 * VIT_CONTEXT_SIZE + (VIT_CONTEXT_SIZE - 1 - currentPosition);

            const int8_t *qHead = q + h * VIT_HEAD_DIM;

            int32_t acc[VIT_CONTEXT_SIZE];

            int32_t accMax = INT32_MIN;

            for (uint32_t j = 0; j < validCount; j += 1) {

                int32_t sum = dotS8S8(qHead, kWindow[layer][h][j], VIT_HEAD_DIM, alibi[j]);

                acc[j] = sum;

                if (sum > accMax) accMax = sum;

            }

            for (uint32_t j = 0; j < validCount; j += 1) {

                int32_t diff = arm_nn_requantize(acc[j] - accMax, params->qkMultiplier, params->qkShift);

                scores[j] = (int8_t)(diff < -128 ? -128 : diff);

            }

        })

        VIT_TIME(softmaxCycles[layer], {

#if VIT_SOFTMAX_LUT

            runSoftmaxTable(scores, validCount, probs);

#else

            arm_softmax_s8(scores, 1, (int32_t)validCount, params->softmaxMultiplier, params->softmaxShift, params->softmaxDiffMin, probs);

#endif

            for (uint32_t j = validCount; j < VIT_CONTEXT_SIZE; j += 1) probs[j] = -128;

        })

        VIT_TIME(avCycles[layer], {

            arm_nn_vec_mat_mult_t_s8(probs, (const int8_t*)vWindowT[layer - 1][h], NULL, NULL, attnOut + h * VIT_HEAD_DIM,
                                     128, 0, params->avMultiplier, params->avShift,
                                     VIT_CONTEXT_SIZE, VIT_HEAD_DIM, -128, 127, 1, 0);

        })

    }

}

#endif

/* LayerNorm1 onwards, shared by every layer: LayerNorm1 on (residual + out_proj's accumulators),
   int8 FFN, LayerNorm2 -> embedding (int8, the next layer's input). The residual is
   residualFloat (layer 0) or, if NULL, embedding at ln1.inputScaleA (next layers). */

static void runLayerTail(uint32_t layer, const int32_t *outProjAcc, const float *residualFloat, int8_t *embedding, ViTInferenceTiming_t *timing) {

    const vit_layer_t *params = &vit_layers[layer];

    int8_t normalised[VIT_EMBED_DIM];

    VIT_TIME(layerNorm1Cycles[layer], runLayerNorm1(&params->ln1, residualFloat, embedding, outProjAcc, params->outProj.accScale, normalised))

    int8_t hidden[VIT_FFN_DIM];

    VIT_TIME(ffn1Cycles[layer], runFullyConnected(&params->ffn1, normalised, VIT_EMBED_DIM, hidden, VIT_FFN_DIM, 0))

    int8_t feedForward[VIT_EMBED_DIM];

    VIT_TIME(ffn2Cycles[layer], runFullyConnected(&params->ffn2, hidden, VIT_FFN_DIM, feedForward, VIT_EMBED_DIM, -128))

    VIT_TIME(layerNorm2Cycles[layer], runLayerNorm2(&params->ln2, normalised, feedForward, embedding))

}

#if VIT_STEM_HISTORY

/* ReLU(Conv2d(1 -> VIT_STEM_CHANNELS)) over the frames in stemHistory, then the patch embedding,
   one channel at a time: the channel's VIT_STEM_OUT_BINS int16 outputs (the patch embedding's
   inputs c * VIT_STEM_OUT_BINS + p) go to a small buffer, then into the VIT_EMBED_DIM
   accumulators (bias included), so only one channel is ever stored. Frequency zero padding
   VIT_STEM_FREQ_PAD on each side: the window then starts or ends inside the bins. At most
   VIT_STEM_FREQ_KERNEL * VIT_STEM_TIME_KERNEL * 32768 * 127 per stem accumulator. */

static void runStemPatchEmbed(int32_t acc[VIT_EMBED_DIM]) {

    for (uint32_t o = 0; o < VIT_EMBED_DIM; o += 1) acc[o] = vit_patch_embed.bias[o];

    for (uint32_t c = 0; c < VIT_STEM_CHANNELS; c += 1) {

        const int8_t *weight = vit_stem.weight + c * VIT_STEM_FREQ_KERNEL * VIT_STEM_TIME_KERNEL;

        int16_t out[VIT_STEM_OUT_BINS];

        for (uint32_t p = 0; p < VIT_STEM_OUT_BINS; p += 1) {

            int32_t first = (int32_t)(p * VIT_STEM_FREQ_STRIDE) - VIT_STEM_FREQ_PAD;

            int32_t fLo = first < 0 ? -first : 0;

            int32_t fHi = first + VIT_STEM_FREQ_KERNEL > VIT_N_BINS ? VIT_N_BINS - first : VIT_STEM_FREQ_KERNEL;

            int32_t sum = dotS8S16(weight + fLo * VIT_STEM_TIME_KERNEL, stemHistory[first + fLo], (fHi - fLo) * VIT_STEM_TIME_KERNEL, vit_stem.bias[c]);

            int32_t value = arm_nn_requantize(sum, vit_stem.multiplier, vit_stem.shift);

            out[p] = (int16_t)(value < 0 ? 0 : (value > 32767 ? 32767 : value)); /* ReLU */

        }

        for (uint32_t o = 0; o < VIT_EMBED_DIM; o += 1) {

            acc[o] = dotS8S16(vit_patch_embed.weight + o * VIT_PATCH_INPUTS + c * VIT_STEM_OUT_BINS, out, VIT_STEM_OUT_BINS, acc[o]);

        }

    }

}

#endif

#if VIT_HAS_STEM && VIT_STEM_BUFFER_ALIGNED

/* The buffer-aligned stem, then the patch embedding: for each frequency position, its
   VIT_STEM_FREQ_KERNEL x 7 window of the buffer's frames is quantized to int16 once (zeros outside
   the bins: the frequency padding) and every channel's ReLU(conv) output is stored; the
   VIT_PATCH_INPUTS outputs (1 KB of stack, no input history kept) then go through the patch
   embedding. */

static void runStemAlignedPatchEmbed(const volatile float spectrogram[VIT_INFERENCE_INPUT_FRAMES][VIT_N_BINS], int32_t acc[VIT_EMBED_DIM]) {

    int16_t out[VIT_STEM_CHANNELS][VIT_STEM_OUT_BINS];

    for (uint32_t p = 0; p < VIT_STEM_OUT_BINS; p += 1) {

        int16_t window[VIT_STEM_FREQ_KERNEL][VIT_STEM_TIME_KERNEL]; /* [freq][time], the weight's layout */

        int32_t first = (int32_t)(p * VIT_STEM_FREQ_STRIDE) - VIT_STEM_FREQ_PAD;

        for (int32_t f = 0; f < VIT_STEM_FREQ_KERNEL; f += 1) {

            int32_t bin = first + f;

            for (uint32_t t = 0; t < VIT_STEM_TIME_KERNEL; t += 1) {

                window[f][t] = (bin < 0 || bin >= VIT_N_BINS) ? 0 : clampToInt16(roundToInt(spectrogram[t][bin] * VIT_INPUT_INV_SCALE));

            }

        }

        for (uint32_t c = 0; c < VIT_STEM_CHANNELS; c += 1) {

            int32_t sum = dotS8S16(vit_stem.weight + c * VIT_STEM_FREQ_KERNEL * VIT_STEM_TIME_KERNEL, &window[0][0],
                                   VIT_STEM_FREQ_KERNEL * VIT_STEM_TIME_KERNEL, vit_stem.bias[c]);

            int32_t value = arm_nn_requantize(sum, vit_stem.multiplier, vit_stem.shift);

            out[c][p] = (int16_t)(value < 0 ? 0 : (value > 32767 ? 32767 : value)); /* ReLU */

        }

    }

    accumulateInt16Input(&vit_patch_embed, &out[0][0], VIT_PATCH_INPUTS, acc, VIT_EMBED_DIM);

}

#endif

#if VIT_CONV_KERNEL

#if VIT_HAS_INT8_LAYERS

/* Convolution module l >= 1, int8: embedding (the previous layer's output) in, the module's
   output out, in place. Depthwise: per channel, a VIT_CONV_KERNEL-tap dot product over its
   history, requantized, ReLU; pointwise: int32 accumulators; residual + LayerNorm in float. */

static void runConvModule(uint32_t layer, int8_t *embedding) {

    const vit_conv_t *conv = &vit_convs[layer];

    int8_t (*history)[VIT_CONV_KERNEL] = convHistory[layer - VIT_CONV0_INT16];

    int8_t hidden[VIT_EMBED_DIM];

    for (uint32_t d = 0; d < VIT_EMBED_DIM; d += 1) {

        for (uint32_t t = 0; t + 1 < VIT_CONV_KERNEL; t += 1) history[d][t] = history[d][t + 1];

        history[d][VIT_CONV_KERNEL - 1] = embedding[d];

        int32_t value = arm_nn_requantize(dotS8S8(conv->depthwise.weight + d * VIT_CONV_KERNEL, history[d], VIT_CONV_KERNEL, conv->depthwise.bias[d]),
                                          conv->depthwise.multiplier, conv->depthwise.shift);

        hidden[d] = (int8_t)(value < 0 ? 0 : (value > 127 ? 127 : value)); /* ReLU */

    }

    int32_t acc[VIT_EMBED_DIM];

    accumulateInt8Input(&conv->pointwise, hidden, VIT_EMBED_DIM, acc, VIT_EMBED_DIM);

    float x[VIT_EMBED_DIM];

    for (uint32_t i = 0; i < VIT_EMBED_DIM; i += 1) x[i] = conv->ln.inputScaleA * (float)embedding[i] + (float)acc[i] * conv->pointwise.accScale;

    normaliseToInt8(&conv->ln, x, embedding);

}

#endif

/* Convolution module 0, between the patch embedding and layer 0: embedding16 / embeddingFloat
   (the patch embedding, int16 and float) in; out, the module's output in the same two forms, as
   layer 0 takes them (int16 at ln.outputInvScale). int16 (VIT_CONV0_INT16): int16 history and
   depthwise output, the residual in float; else int8 throughout, the residual from the int8 input. */

static void runConvModule0(int16_t *embedding16, float *embeddingFloat) {

    const vit_conv_t *conv = &vit_convs[0];

    int32_t acc[VIT_EMBED_DIM];

    float x[VIT_EMBED_DIM];

#if VIT_CONV0_INT16

    int16_t hidden[VIT_EMBED_DIM];

    for (uint32_t d = 0; d < VIT_EMBED_DIM; d += 1) {

        for (uint32_t t = 0; t + 1 < VIT_CONV_KERNEL; t += 1) convHistory16[d][t] = convHistory16[d][t + 1];

        convHistory16[d][VIT_CONV_KERNEL - 1] = embedding16[d];

        int32_t value = arm_nn_requantize(dotS8S16(conv->depthwise.weight + d * VIT_CONV_KERNEL, convHistory16[d], VIT_CONV_KERNEL, conv->depthwise.bias[d]),
                                          conv->depthwise.multiplier, conv->depthwise.shift);

        hidden[d] = (int16_t)(value < 0 ? 0 : (value > 32767 ? 32767 : value)); /* ReLU */

    }

    accumulateInt16Input(&conv->pointwise, hidden, VIT_EMBED_DIM, acc, VIT_EMBED_DIM);

    for (uint32_t i = 0; i < VIT_EMBED_DIM; i += 1) x[i] = embeddingFloat[i] + (float)acc[i] * conv->pointwise.accScale;

#else

    int8_t input[VIT_EMBED_DIM], hidden[VIT_EMBED_DIM];

    for (uint32_t d = 0; d < VIT_EMBED_DIM; d += 1) {

        input[d] = clampToInt8(roundToInt(embeddingFloat[d] * VIT_CONV0_IN_INV_SCALE));

        for (uint32_t t = 0; t + 1 < VIT_CONV_KERNEL; t += 1) convHistory[0][d][t] = convHistory[0][d][t + 1];

        convHistory[0][d][VIT_CONV_KERNEL - 1] = input[d];

        int32_t value = arm_nn_requantize(dotS8S8(conv->depthwise.weight + d * VIT_CONV_KERNEL, convHistory[0][d], VIT_CONV_KERNEL, conv->depthwise.bias[d]),
                                          conv->depthwise.multiplier, conv->depthwise.shift);

        hidden[d] = (int8_t)(value < 0 ? 0 : (value > 127 ? 127 : value)); /* ReLU */

    }

    accumulateInt8Input(&conv->pointwise, hidden, VIT_EMBED_DIM, acc, VIT_EMBED_DIM);

    for (uint32_t i = 0; i < VIT_EMBED_DIM; i += 1) x[i] = conv->ln.inputScaleA * (float)input[i] + (float)acc[i] * conv->pointwise.accScale;

#endif

    layerNormFloat(&conv->ln, x);

    for (uint32_t i = 0; i < VIT_EMBED_DIM; i += 1) {

        embeddingFloat[i] = x[i];

        embedding16[i] = clampToInt16(roundToInt(x[i] * conv->ln.outputInvScale));

    }

}

#endif

/* Layer 0: embedding16 (the patch embedding in int16) for Q/K/V, embeddingFloat (the same, in
   float) as LayerNorm1's residual; writes the layer's int8 output to embedding. */

static void runLayer0(const int16_t *embedding16, const float *embeddingFloat, int8_t *embedding, ViTInferenceTiming_t *timing) {

    const vit_layer_t *params = &vit_layers[0];

    int16_t q[VIT_EMBED_DIM], v[VIT_EMBED_DIM];

    int8_t k[VIT_EMBED_DIM];

    VIT_TIME(qkvCycles[0], {

        runFullyConnectedInt16(&params->q, embedding16, VIT_EMBED_DIM, q, VIT_EMBED_DIM);

        runFullyConnectedInt16ToInt8(&params->k, embedding16, VIT_EMBED_DIM, k, VIT_EMBED_DIM);

        runFullyConnectedInt16(&params->v, embedding16, VIT_EMBED_DIM, v, VIT_EMBED_DIM);

        for (uint32_t h = 0; h < VIT_N_HEADS; h += 1) {

            for (uint32_t i = 0; i < VIT_HEAD_DIM; i += 1) {

                kWindow[0][h][currentPosition][i] = k[h * VIT_HEAD_DIM + i];

                vWindowT16[h][i][currentPosition] = v[h * VIT_HEAD_DIM + i];

            }

        }

    })

    int16_t attnOut[VIT_EMBED_DIM];

    runAttentionInt16(q, attnOut, timing);

    int32_t outProjAcc[VIT_EMBED_DIM];

    VIT_TIME(outProjCycles[0], accumulateInt16Input(&params->outProj, attnOut, VIT_EMBED_DIM, outProjAcc, VIT_EMBED_DIM))

    runLayerTail(0, outProjAcc, embeddingFloat, embedding, timing);

}

#if VIT_HAS_INT8_LAYERS

/* Layers >= 1: embedding is the int8 input (Q/K/V and, at ln1.inputScaleA, the residual),
   overwritten with the layer's output. */

static void runLayer(uint32_t layer, int8_t *embedding, ViTInferenceTiming_t *timing) {

    const vit_layer_t *params = &vit_layers[layer];

    int8_t q[VIT_EMBED_DIM], k[VIT_EMBED_DIM], v[VIT_EMBED_DIM];

    VIT_TIME(qkvCycles[layer], {

        runFullyConnected(&params->q, embedding, VIT_EMBED_DIM, q, VIT_EMBED_DIM, -128);

        runFullyConnected(&params->k, embedding, VIT_EMBED_DIM, k, VIT_EMBED_DIM, -128);

        runFullyConnected(&params->v, embedding, VIT_EMBED_DIM, v, VIT_EMBED_DIM, -128);

        for (uint32_t h = 0; h < VIT_N_HEADS; h += 1) {

            for (uint32_t i = 0; i < VIT_HEAD_DIM; i += 1) {

                kWindow[layer][h][currentPosition][i] = k[h * VIT_HEAD_DIM + i];

                vWindowT[layer - 1][h][i][currentPosition] = v[h * VIT_HEAD_DIM + i];

            }

        }

    })

    int8_t attnOut[VIT_EMBED_DIM];

    runAttention(layer, q, attnOut, timing);

    int32_t outProjAcc[VIT_EMBED_DIM];

    VIT_TIME(outProjCycles[layer], accumulateInt8Input(&params->outProj, attnOut, VIT_EMBED_DIM, outProjAcc, VIT_EMBED_DIM))

    runLayerTail(layer, outProjAcc, NULL, embedding, timing);

}

#endif

/* One token from the patch embedding's int32 accumulators (bias included): the layers, the
   classifier -> tokenLogits. Advances the K/V ring. */

static void runToken(const int32_t acc[VIT_EMBED_DIM], ViTInferenceTiming_t *timing, float tokenLogits[VIT_N_CLASSES]) {

    currentPosition = (currentPosition + 1) % VIT_CONTEXT_SIZE;

    if (validCount < VIT_CONTEXT_SIZE) validCount += 1;

    int16_t embedding16[VIT_EMBED_DIM];

    float embeddingFloat[VIT_EMBED_DIM];

    VIT_TIME(patchEmbedCycles, {

        /* one accumulation: float for layer 0's residual, int16 for its Q/K/V */

        for (uint32_t i = 0; i < VIT_EMBED_DIM; i += 1) {

            embeddingFloat[i] = (float)acc[i] * vit_patch_embed.accScale;

            embedding16[i] = clampToInt16(roundToInt(embeddingFloat[i] * VIT_EMBED_INV_SCALE));

        }

    })

#if VIT_CONV_KERNEL

    VIT_TIME(convCycles[0], runConvModule0(embedding16, embeddingFloat))

#endif

    int8_t embedding[VIT_EMBED_DIM];

    runLayer0(embedding16, embeddingFloat, embedding, timing);

#if VIT_HAS_INT8_LAYERS

    for (uint32_t layer = 1; layer < VIT_DEPTH; layer += 1) {

#if VIT_CONV_KERNEL

        VIT_TIME(convCycles[layer], runConvModule(layer, embedding))

#endif

        runLayer(layer, embedding, timing);

    }

#endif

    VIT_TIME(classifierCycles, {

        for (uint32_t c = 0; c < VIT_N_CLASSES; c += 1) {

            const int8_t *weight = vit_classifier_weight + c * VIT_EMBED_DIM;

            int32_t acc = 0;

            for (uint32_t i = 0; i < VIT_EMBED_DIM; i += 1) acc += (int32_t)weight[i] * (int32_t)embedding[i];

            tokenLogits[c] = (float)acc * VIT_CLASSIFIER_LOGIT_SCALE + vit_classifier_bias[c];

        }

    })

    timing->tokensDone += 1;

}

void ViTInference_run(const volatile float spectrogram[VIT_INFERENCE_INPUT_FRAMES][VIT_N_BINS], ViTInferenceTiming_t *timing, uint32_t deadlineCycles, float logits[VIT_N_CLASSES]) {

    memset(timing, 0, sizeof(ViTInferenceTiming_t));

    float logitSum[VIT_N_CLASSES] = { 0.0f };

    uint32_t lastTokenCycles = 0;

#if VIT_HAS_STEM && VIT_STEM_BUFFER_ALIGNED

    /* one token for the whole buffer (no per-token cost estimate: only an already-passed deadline skips it) */

    (void)lastTokenCycles;

    if (deadlineCycles != 0 && (int32_t)(deadlineCycles - VIT_GET_CYCLES()) < 0) {

        for (uint32_t c = 0; c < VIT_N_CLASSES; c += 1) logits[c] = 0.0f;

        return;

    }

    int32_t acc[VIT_EMBED_DIM];

    VIT_TIME(stemCycles, runStemAlignedPatchEmbed(spectrogram, acc))

    runToken(acc, timing, logits);

    timing->framesDone = VIT_INFERENCE_INPUT_FRAMES;

    (void)logitSum;

#else

    for (uint32_t frame = 0; frame < VIT_INFERENCE_INPUT_FRAMES; frame += 1) {

        uint32_t frameStartCycles = VIT_GET_CYCLES();

#if VIT_STEM_HISTORY

        int runsToken = frameCount % VIT_STEM_TIME_STRIDE == 0;

#else

        int runsToken = 1;

#endif

        /* a frame that would start a token ending after the deadline (at the previous token's cost) is not run */

        if (runsToken && deadlineCycles != 0 && (int32_t)(deadlineCycles - (frameStartCycles + lastTokenCycles)) < 0) break;

#if VIT_STEM_HISTORY

        VIT_TIME(stemCycles, {

            for (uint32_t bin = 0; bin < VIT_N_BINS; bin += 1) {

                int16_t *history = stemHistory[bin];

                for (uint32_t t = 0; t + 1 < VIT_STEM_TIME_KERNEL; t += 1) history[t] = history[t + 1];

                history[VIT_STEM_TIME_KERNEL - 1] = clampToInt16(roundToInt(spectrogram[frame][bin] * VIT_INPUT_INV_SCALE));

            }

        })

        frameCount += 1;

        if (runsToken) {

            int32_t acc[VIT_EMBED_DIM];

            VIT_TIME(stemCycles, runStemPatchEmbed(acc))

            runToken(acc, timing, lastTokenLogits);

            lastTokenCycles = VIT_GET_CYCLES() - frameStartCycles;

        }

        for (uint32_t c = 0; c < VIT_N_CLASSES; c += 1) logitSum[c] += lastTokenLogits[c];

#else

        int32_t acc[VIT_EMBED_DIM];

        VIT_TIME(patchEmbedCycles, {

            int16_t input[VIT_N_BINS];

            for (uint32_t bin = 0; bin < VIT_N_BINS; bin += 1) input[bin] = clampToInt16(roundToInt(spectrogram[frame][bin] * VIT_INPUT_INV_SCALE));

            accumulateInt16Input(&vit_patch_embed, input, VIT_N_BINS, acc, VIT_EMBED_DIM);

        })

        float tokenLogits[VIT_N_CLASSES];

        runToken(acc, timing, tokenLogits);

        lastTokenCycles = VIT_GET_CYCLES() - frameStartCycles;

        for (uint32_t c = 0; c < VIT_N_CLASSES; c += 1) logitSum[c] += tokenLogits[c];

#endif

        timing->framesDone += 1;

    }

    for (uint32_t c = 0; c < VIT_N_CLASSES; c += 1) logits[c] = timing->framesDone == 0 ? 0.0f : logitSum[c] / (float)timing->framesDone;

#endif

}
