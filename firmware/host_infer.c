/****************************************************************************
 * host_infer.c
 * The AudioMoth's Conformer (frontend.c + conformer_inference.c + an export of export.py) on the host
 *****************************************************************************/

/* Streams every recording buffer by buffer exactly as the device, the front end's PCEN state and
   the Conformer's history reset at the start of each recording, and writes every buffer's logits.

   host_infer AUDIO_IN LOGITS_OUT PCEN_INIT GAIN
     AUDIO_IN  : int32 N, int32 S (samples per recording, a multiple of 1024), int16 [N][S]
     LOGITS_OUT: int32 N, int32 B (buffers per recording), int32 C (classes), float32 [N][B][C]
     PCEN_INIT : 0 = model_pcen_initial_m (the device's seed), 1 = the first frame
     GAIN      : input gain (1 for int16 WAV data, 16 for the device's raw ADC samples) */

#include <stdio.h>
#include <stdlib.h>

#include "frontend.h"
#include "conformer_inference.h"

#if CONF_N_BINS != FRONTEND_NUMBER_OF_BINS || CONF_INFERENCE_INPUT_FRAMES != FRONTEND_NUMBER_OF_FRAMES
#error "conformer_model.h does not match the front end"
#endif

static void fail(const char *message) {

    fprintf(stderr, "host_infer: %s\n", message);

    exit(1);

}

int main(int argc, char **argv) {

    if (argc != 5) fail("usage: host_infer AUDIO_IN LOGITS_OUT PCEN_INIT GAIN");

    FILE *in = fopen(argv[1], "rb"), *out = fopen(argv[2], "wb");

    if (!in || !out) fail("cannot open the input or the output");

    int pcenInit = atoi(argv[3]);

    float gain = (float)atof(argv[4]);

    int32_t header[3];

    if (fread(header, sizeof(int32_t), 2, in) != 2) fail("short read");

    int32_t recordings = header[0], samples = header[1];

    if (samples % FRONTEND_BUFFER_SAMPLES != 0) fail("samples per recording must be a multiple of 1024");

    int32_t buffers = samples / FRONTEND_BUFFER_SAMPLES;

    header[1] = buffers;

    header[2] = CONF_N_CLASSES;

    fwrite(header, sizeof(int32_t), 3, out);

    int16_t *audio = malloc(sizeof(int16_t) * samples);

    float spectrogram[FRONTEND_NUMBER_OF_FRAMES][FRONTEND_NUMBER_OF_BINS];

    float logits[CONF_N_CLASSES];

    ConformerInferenceTiming_t timing;

    Frontend_initialise();

    for (int32_t n = 0; n < recordings; n += 1) {

        if (fread(audio, sizeof(int16_t), samples, in) != (size_t)samples) fail("short read");

        Frontend_reset(pcenInit);

        ConformerInference_reset();

        for (int32_t b = 0; b < buffers; b += 1) {

            Frontend_computeSpectrogram(audio + b * FRONTEND_BUFFER_SAMPLES, gain, spectrogram);

            ConformerInference_run(spectrogram, &timing, 0, logits);

            fwrite(logits, sizeof(float), CONF_N_CLASSES, out);

        }

    }

    free(audio);

    fclose(in);

    fclose(out);

    return 0;

}
