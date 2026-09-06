// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Hint: measure after every change. Not every "optimization" helps  let the numbers,
// not intuition, decide.

#include <immintrin.h>
#include "convolution.h"

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {

    const int p = K / 2;
    const int in_stride = W + 2 * p;

    int m = 0;
    int stride1, stride2;
    __m256 c1,a1;
    __m256 c2,a2;
    __m256 c3,a3;
    __m256 c4,a4;
    __m256 b;

    int oy;
    for (oy = 0; oy + 3 < H; oy += 4) {

        for (int ox = 0; ox < W; ox += 8) {

            c1 = _mm256_setzero_ps();
            c2 = _mm256_setzero_ps();
            c3 = _mm256_setzero_ps();
            c4 = _mm256_setzero_ps();

            for (int j = 0; j < K; j++) {

                stride2 = j * K;

                for (int i = 0; i < K; i++) {

                    b = _mm256_set1_ps(ker[stride2 + i]);

                    stride1 = (oy + j) * in_stride;
                    a1 = _mm256_load_ps(&in[stride1 + ox + i]);

                    stride1 = (oy + 1 + j) * in_stride;
                    a2 = _mm256_load_ps(&in[stride1 + ox + i]);

                    stride1 = (oy + 2 + j) * in_stride;
                    a3 = _mm256_load_ps(&in[stride1 + ox + i]);

                    stride1 = (oy + 3 + j) * in_stride;
                    a4 = _mm256_load_ps(&in[stride1 + ox + i]);

                    c1 = _mm256_fmadd_ps(a1, b, c1);
                    c2 = _mm256_fmadd_ps(a2, b, c2);
                    c3 = _mm256_fmadd_ps(a3, b, c3);
                    c4 = _mm256_fmadd_ps(a4, b, c4);
                }
            }

            m = oy * W + ox;

            _mm256_store_ps(&out[m], c1);
            _mm256_store_ps(&out[m + W], c2);
            _mm256_store_ps(&out[m + 2 * W], c3);
            _mm256_store_ps(&out[m + 3 * W], c4);
        }
    }

    // remaining rows
    for (; oy < H; oy++) {

        for (int ox = 0; ox < W; ox += 8) {

            c1 = _mm256_setzero_ps();

            for (int j = 0; j < K; j++) {

                stride1 = (oy + j) * in_stride;
                stride2 = j * K;

                for (int i = 0; i < K; i++) {

                    a1 = _mm256_load_ps(&in[stride1 + ox + i]);
                    b = _mm256_set1_ps(ker[stride2 + i]);
                    c1 = _mm256_fmadd_ps(a1, b, c1);
                }
            }

            _mm256_store_ps(&out[oy * W + ox], c1);
        }
    }
}