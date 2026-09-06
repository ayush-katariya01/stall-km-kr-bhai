// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
#include "convolution.h"

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K)
{
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < H; ++oy){
        int ox = 0;

        for (; ox + 3 < W; ox += 4){
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;

            for (int ky = 0; ky < K; ++ky){
                for (int kx = 0; kx < K; ++kx){
                    float kval = ker[ky * K + kx];

                    const float* in_ptr =
                        &in[(oy + ky) * in_stride + ox + kx];

                    acc0 += in_ptr[0] * kval;
                    acc1 += in_ptr[1] * kval;
                    acc2 += in_ptr[2] * kval;
                    acc3 += in_ptr[3] * kval;
                }
            }

            out[oy * W + ox]     = acc0;
            out[oy * W + ox + 1] = acc1;
            out[oy * W + ox + 2] = acc2;
            out[oy * W + ox + 3] = acc3;
        }

        for (; ox < W; ++ox){
            float acc = 0.0f;
            for (int ky = 0; ky < K; ++ky){
                for (int kx = 0; kx < K; ++kx){
                    acc += in[(oy + ky) * in_stride + (ox + kx)]
                         * ker[ky * K + kx];
                }
            }
            out[oy * W + ox] = acc;
        }
    }
}