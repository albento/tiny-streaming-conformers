/****************************************************************************
 * conformer_inference.h
 * openacousticdevices.info
 *****************************************************************************/

#ifndef __CONF_INFERENCE_H
#define __CONF_INFERENCE_H

#include <stdint.h>

#include "conformer_model.h"


/* Per-stage CPU-cycle breakdown (DWT->CYCCNT on the device, 0 on a host build) of the last
   ConformerInference_run() call, summed over the buffer's CONF_INFERENCE_INPUT_FRAMES tokens. */

typedef struct {
    uint32_t stemCycles;                        /* stem: input int16 quantization + convolution fused with patch_embed (0 without a stem) */
    uint32_t patchEmbedCycles;                  /* no stem: input int16 quantization + patch_embed; both: its int16/float outputs */
    uint32_t qkvCycles[CONF_DEPTH];              /* q/k/v projections + K/V ring buffer write */
    uint32_t qkCycles[CONF_DEPTH];               /* Q.K^T + ALiBi + row max + diff requantization */
    uint32_t softmaxCycles[CONF_DEPTH];          /* arm_softmax_s8 */
    uint32_t avCycles[CONF_DEPTH];               /* attn.V */
    uint32_t outProjCycles[CONF_DEPTH];
    uint32_t layerNorm1Cycles[CONF_DEPTH];       /* residual add + LayerNorm, float */
    uint32_t ffn1Cycles[CONF_DEPTH];             /* linear1 + fused ReLU */
    uint32_t ffn2Cycles[CONF_DEPTH];
    uint32_t layerNorm2Cycles[CONF_DEPTH];
    uint32_t convCycles[CONF_DEPTH];             /* convolution module before each layer (CONF_CONV_KERNEL > 0, else 0) */
    uint32_t classifierCycles;
    uint32_t tokensDone;                        /* tokens run (one per frame, or per CONF_STEM_TIME_STRIDE frames with a stem) */
    uint32_t framesDone;                        /* < CONF_INFERENCE_INPUT_FRAMES if the deadline cut the buffer short */
} ConformerInferenceTiming_t;

/* Clears the K/V history - call once before the first buffer of a recording. */

void ConformerInference_reset(void);

/* Streams one buffer's CONF_INFERENCE_INPUT_FRAMES PCEN frames through the model, one token per
   frame (per CONF_STEM_TIME_STRIDE frames with a stem), and writes the buffer's CONF_N_CLASSES
   logits: the mean over its frames of the latest token's logits, as conformer.py's Conformer.forward()
   averages them. spectrogram has main.c's stftSpectrogramDb layout
   [frame][bin]. Carries K/V history across calls. timing must not be NULL.

   deadlineCycles: absolute DWT->CYCCNT value; if non-zero, no token is started that would,
   at the previous token's cost, end after it - the remaining frames are skipped (see
   timing->framesDone) and the logits are the mean over the frames actually run. 0 = no limit. */

void ConformerInference_run(const volatile float spectrogram[CONF_INFERENCE_INPUT_FRAMES][CONF_N_BINS],
                      ConformerInferenceTiming_t *timing, uint32_t deadlineCycles, float logits[CONF_N_CLASSES]);

#endif
