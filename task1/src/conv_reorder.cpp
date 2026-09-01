// conv_reorder.cpp
// STAGE 1: LOOP REORDERING

#include "convolution.h"

void conv_reorder(const float* in, float* out, const float* ker,
                  int H, int W, int K) {

    const int p = K / 2;
    const int in_stride = W + 2 * p;

    // Since we will keep adding to the output,
    // first initialize every output element to 0.
    for (int i = 0; i < H * W; i++) {
        out[i] = 0.0f;
    }

    // Reordered loops
    for (int ky = 0; ky < K; ky++) {
        for (int kx = 0; kx < K; kx++) {
            
            // Load one kernel value once
            float kval = ker[ky * K + kx];

            for (int oy = 0; oy < H; oy++) {
                const float *in_row = &in[(oy + ky) * in_stride + kx];
                float *out_row = &out[oy * W];

                for (int ox = 0; ox < W; ox++) {

                    out_row[ox] += in_row[ox] * kval;
                }
            }
        }
    }
}