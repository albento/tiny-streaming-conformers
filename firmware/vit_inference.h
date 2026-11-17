/****************************************************************************
 * vit_inference.h
 * openacousticdevices.info
 *****************************************************************************/

#ifndef __VIT_INFERENCE_H
#define __VIT_INFERENCE_H

#include <stdint.h>

#include "vit_model.h"


/* Per-stage CPU-cycle breakdown (DWT->CYCCNT on the device, 0 on a host build) of the last
   ViTInference_run() call, summed over the buffer's VIT_INFERENCE_INPUT_FRAMES tokens. */

typedef struct {
    uint32_t stemCycles;                        /* stem: input int16 quantization + convolution fused with patch_embed (0 without a stem) */
    uint32_t patchEmbedCycles;                  /* no stem: input int16 quantization + patch_embed; both: its int16/float outputs */
    uint32_t qkvCycles[VIT_DEPTH];              /* q/k/v projections + K/V ring buffer write */
    uint32_t qkCycles[VIT_DEPTH];               /* Q.K^T + ALiBi + row max + diff requantization */
    uint32_t softmaxCycles[VIT_DEPTH];          /* arm_softmax_s8 */
    uint32_t avCycles[VIT_DEPTH];               /* attn.V */
    uint32_t outProjCycles[VIT_DEPTH];
    uint32_t layerNorm1Cycles[VIT_DEPTH];       /* residual add + LayerNorm, float */
    uint32_t ffn1Cycles[VIT_DEPTH];             /* linear1 + fused ReLU */
    uint32_t ffn2Cycles[VIT_DEPTH];
    uint32_t layerNorm2Cycles[VIT_DEPTH];
    uint32_t convCycles[VIT_DEPTH];             /* convolution module before each layer (VIT_CONV_KERNEL > 0, else 0) */
    uint32_t classifierCycles;
    uint32_t tokensDone;                        /* tokens run (one per frame, or per VIT_STEM_TIME_STRIDE frames with a stem) */
    uint32_t framesDone;                        /* < VIT_INFERENCE_INPUT_FRAMES if the deadline cut the buffer short */
} ViTInferenceTiming_t;

/* Clears the K/V history - call once before the first buffer of a recording. */

void ViTInference_reset(void);

/* Streams one buffer's VIT_INFERENCE_INPUT_FRAMES PCEN frames through the model, one token per
   frame (per VIT_STEM_TIME_STRIDE frames with a stem), and writes the buffer's VIT_N_CLASSES
   logits: the mean over its frames of the latest token's logits, as vit.py's ViT.forward()
   averages them. spectrogram has main.c's stftSpectrogramDb layout
   [frame][bin]. Carries K/V history across calls. timing must not be NULL.

   deadlineCycles: absolute DWT->CYCCNT value; if non-zero, no token is started that would,
   at the previous token's cost, end after it - the remaining frames are skipped (see
   timing->framesDone) and the logits are the mean over the frames actually run. 0 = no limit. */

void ViTInference_run(const volatile float spectrogram[VIT_INFERENCE_INPUT_FRAMES][VIT_N_BINS],
                      ViTInferenceTiming_t *timing, uint32_t deadlineCycles, float logits[VIT_N_CLASSES]);

#endif
