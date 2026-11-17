/****************************************************************************
 * frontend.h
 * The ViT's front end on the AudioMoth and on the host: STFT + learned PCEN
 *****************************************************************************/

#ifndef __FRONTEND_H
#define __FRONTEND_H

#include <stdint.h>

#include "pcen.h"

/* One 1024-sample buffer (64 ms at 16 kHz) -> 7 frames of 129 bins: 256-point FFT, hop 128,
   restarted at every buffer (no frame spans two buffers) */

#define FRONTEND_BUFFER_SAMPLES                  1024
#define FRONTEND_FFT_LENGTH                      256
#define FRONTEND_HOP_LENGTH                      128
#define FRONTEND_NUMBER_OF_FRAMES                ((FRONTEND_BUFFER_SAMPLES - FRONTEND_FFT_LENGTH) / FRONTEND_HOP_LENGTH + 1)
#define FRONTEND_NUMBER_OF_BINS                  (FRONTEND_FFT_LENGTH / 2 + 1)

/* PCEN's running mean M at the start of a recording: model_pcen_initial_m (the mean training
   power per bin, what the device uses), or the first frame's power */

#define FRONTEND_PCEN_INIT_TRAINING_MEAN         0
#define FRONTEND_PCEN_INIT_FIRST_FRAME           1

void Frontend_initialise(void);

/* Call before the first buffer of every recording. */

void Frontend_reset(int pcenInitMode);

/* One buffer -> spectrogram[frame][bin] (PCEN output), carrying M across calls. inputGain scales
   the samples first (16 on the device for its raw ADC samples, 1 for int16 WAV data). */

void Frontend_computeSpectrogram(const int16_t *source, float inputGain, float spectrogram[FRONTEND_NUMBER_OF_FRAMES][FRONTEND_NUMBER_OF_BINS]);

#endif
