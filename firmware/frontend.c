/****************************************************************************
 * frontend.c
 * The Conformer's front end on the AudioMoth and on the host: STFT + learned PCEN
 *****************************************************************************/

/* Per buffer: Hann window (constant table, stft_window.h), CMSIS-DSP's arm_rfft_fast_f32, magnitude
   with the bins below 150 Hz zeroed, then per bin PCEN with the exported parameters (pcen.c):
   (E / (M + eps)^alpha + delta)^r - delta^r, E the power, M its running mean. The two fractional
   powers use stftFastPow, an IEEE-754 bit trick (export.py reproduces it for delta^r, so that a
   silent bin gives exactly 0); conformer.py's PCEN uses exact powers in training. */

#include <math.h>
#include <string.h>

#include "arm_math.h"

#include "frontend.h"
#include "pcen.h"
#include "stft_window.h"

#if MODEL_PCEN_NUMBER_OF_BINS != FRONTEND_NUMBER_OF_BINS
#error "pcen.h does not match the STFT bin count"
#endif

#define STFT_HIGHPASS_BIN                        ((150 * FRONTEND_FFT_LENGTH + 15999) / 16000)  /* first bin >= 150 Hz at 16 kHz */


#define STFT_LOG2_BIAS_Q16_16                    8320252  /* round(126.95696433397204 * 65536) */

static arm_rfft_fast_instance_f32 stftFFTInstance;

static float32_t stftFrame[FRONTEND_FFT_LENGTH];

static float32_t stftFFTOutput[FRONTEND_FFT_LENGTH];

static float32_t stftMagnitude[FRONTEND_NUMBER_OF_BINS];

static float pcenRunningM[FRONTEND_NUMBER_OF_BINS];

static int pcenSeedFromFirstFrame;

static inline int32_t stftFastLog2Q16_16(float32_t x) {

    int32_t i;

    memcpy(&i, &x, sizeof(i));

    return (i >> 7) - STFT_LOG2_BIAS_Q16_16;

}

static inline float32_t stftFastExp2Q16_16(int32_t log2Fixed) {

    int32_t i = (log2Fixed + STFT_LOG2_BIAS_Q16_16) << 7;

    float32_t result;

    memcpy(&result, &i, sizeof(i));

    return result;

}

static inline float32_t stftFastPow(float32_t x, float32_t a) {

    int32_t log2Fixed = stftFastLog2Q16_16(x);

    int32_t scaledLog2Fixed = (int32_t)((float32_t)log2Fixed * a);

    return stftFastExp2Q16_16(scaledLog2Fixed);

}

static inline float32_t computePcenBin(uint32_t bin, float32_t magnitude) {

    float32_t power = magnitude * magnitude;

    if (pcenSeedFromFirstFrame) {

        pcenRunningM[bin] = power;

    } else {

        pcenRunningM[bin] = model_pcen_ema_alpha[bin] * power + (1.0f - model_pcen_ema_alpha[bin]) * pcenRunningM[bin];

    }

    float32_t denom = stftFastPow(pcenRunningM[bin] + MODEL_PCEN_EPS, model_pcen_alpha[bin]);

    float32_t ratio = power / denom;

    float32_t base = ratio + model_pcen_delta[bin];

    return stftFastPow(base, model_pcen_r[bin]) - model_pcen_delta_pow_r[bin];

}

void Frontend_initialise(void) {

    arm_rfft_fast_init_256_f32(&stftFFTInstance);  /* only the 256-point tables: the general init links them all */

}

void Frontend_reset(int pcenInitMode) {

    for (uint32_t bin = 0; bin < FRONTEND_NUMBER_OF_BINS; bin += 1) {

        pcenRunningM[bin] = model_pcen_initial_m[bin];

    }

    pcenSeedFromFirstFrame = pcenInitMode == FRONTEND_PCEN_INIT_FIRST_FRAME;

}

void Frontend_computeSpectrogram(const int16_t *source, float inputGain, float spectrogram[FRONTEND_NUMBER_OF_FRAMES][FRONTEND_NUMBER_OF_BINS]) {

    float bufferGain = inputGain;

    const int16_t *input = source;

    for (uint32_t frame = 0; frame < FRONTEND_NUMBER_OF_FRAMES; frame += 1) {

        uint32_t offset = frame * FRONTEND_HOP_LENGTH;

        for (uint32_t i = 0; i < FRONTEND_FFT_LENGTH; i += 1) {

            stftFrame[i] = (float32_t)input[offset + i] * stftWindow256[i] * bufferGain;

        }

        arm_rfft_fast_f32(&stftFFTInstance, stftFrame, stftFFTOutput, 0);

        /* CMSIS-DSP packs the real DC and Nyquist bins into the first complex slot of the output */

        stftMagnitude[0] = 0.0f;

        stftMagnitude[FRONTEND_FFT_LENGTH / 2] = fabsf(stftFFTOutput[1]);

        for (uint32_t bin = 1; bin < FRONTEND_FFT_LENGTH / 2; bin += 1) {

            float32_t re = stftFFTOutput[2 * bin];

            float32_t im = stftFFTOutput[2 * bin + 1];

            stftMagnitude[bin] = (bin < STFT_HIGHPASS_BIN) ? 0.0f : sqrtf(re * re + im * im);

        }

        for (uint32_t bin = 0; bin < FRONTEND_NUMBER_OF_BINS; bin += 1) spectrogram[frame][bin] = computePcenBin(bin, stftMagnitude[bin]);

        /* The first-frame seed only applies to frame 0 of the clip */

        pcenSeedFromFirstFrame = 0;

    }

}
