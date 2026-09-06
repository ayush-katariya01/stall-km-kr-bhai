#include "convolution.h"

void conv_reorder(const float* in, float* out, const float* ker,
                  int H, int W, int K)
{
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int i = 0; i < H * W; ++i)
        out[i] = 0.0f;

    for (int ky = 0; ky < K; ++ky){
        const float* ker_row = ker + ky * K;

        for (int oy = 0; oy < H; ++oy){
            float* out_row = out + oy * W;
            const float* in_base = in + (oy + ky) * in_stride;

            for (int kx = 0; kx < K; ++kx){
                const float kval = ker_row[kx];
                const float* in_row = in_base + kx;

                for (int ox = 0; ox < W; ++ox){
                    out_row[ox] += in_row[ox] * kval;
                }
            }
        }
    }
}


