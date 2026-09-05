// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Hint: measure after every change. Not every "optimization" helps  let the numbers,
// not intuition, decide.

#include <immintrin.h>
#include "convolution.h"

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    // TODO(student): replace this placeholder with your best combined implementation.
    //conv_naive(in, out, ker, H, W, K);

    const int p = K / 2;
    const int in_stride = W + 2 * p;

    int b = 192;

    for (int oy = 0; oy < H; oy = oy + b) {
        for (int ox = 0; ox < W; ox = ox + b) {

            int input_tile_end_i = (oy + b < H) ? (oy + b) : H;
            int input_tile_end_j = (ox + b < W) ? (ox + b) : W;

            for (int oyy = oy; oyy < input_tile_end_i; oyy++) {

                for (int oxx = ox; oxx < input_tile_end_j; oxx = oxx + 8) {

                    __m256 c = _mm256_setzero_ps();

                    for (int ky = 0; ky < K; ky++) {

                        int stride1 = (oyy + ky) * in_stride;
                        int stride2 = ky * K;

                        for (int kx = 0; kx < K; kx++) {

                            __m256 a =_mm256_loadu_ps(&in[stride1 + oxx + kx]);
                            __m256 w =_mm256_set1_ps(ker[stride2 + kx]);

                            c = _mm256_fmadd_ps(a, w, c);
                        }
                    }

                    _mm256_storeu_ps(&out[oyy * W + oxx], c);
                }
            }
        }
    }
}