// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Hint: measure after every change. Not every "optimization" helps -- let the numbers,
// not intuition, decide.

#include <immintrin.h>
#include "convolution.h"

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {

    const int p = K / 2;
    const int in_stride = W + 2 * p;

    int m1 = 0 , m2 = 0, m3 = 0, m4 = 0;
    int stride1_a,stride1_b,stride1_c,stride1_d; 
    int stride2;

    __m256 c1, a1;
    __m256 c2, a2;
    __m256 c3, a3;
    __m256 c4, a4;
    __m256 b;

    const int tile_h = 88;
    const int tile_w = 272;

    int oy;

    for (oy = 0; oy < H; oy += tile_h) {

        for (int ox = 0; ox < W; ox += tile_w) {

            int tile_end_y = (oy + tile_h < H) ? (oy + tile_h) : H;
            int tile_end_x = (ox + tile_w < W) ? (ox + tile_w) : W;

            int ty;

            for (ty = oy; ty + 3 < tile_end_y; ty += 4) {

                for (int tx = ox; tx + 7 < tile_end_x; tx += 8) {

                    c1 = _mm256_setzero_ps();
                    c2 = _mm256_setzero_ps();
                    c3 = _mm256_setzero_ps();
                    c4 = _mm256_setzero_ps();

                    for (int j = 0; j < K; j++) {

                        stride2 = j * K;

                        stride1_a = (ty + j) * in_stride;
                        stride1_b = stride1_a + in_stride;
                        stride1_c = stride1_b + in_stride;
                        stride1_d = stride1_c + in_stride; 
                        
                        for (int i = 0; i < K; i++) {

                            b =  _mm256_set1_ps(ker[stride2 + i]);

                            a1 = _mm256_load_ps(in + stride1_a + tx + i);
                            a2 = _mm256_load_ps(in + stride1_b + tx + i);
                            a3 = _mm256_load_ps(in + stride1_c + tx + i);
                            a4 = _mm256_load_ps(in + stride1_d + tx + i);

                            c1 = _mm256_fmadd_ps(a1, b, c1);
                            c2 = _mm256_fmadd_ps(a2, b, c2);
                            c3 = _mm256_fmadd_ps(a3, b, c3);
                            c4 = _mm256_fmadd_ps(a4, b, c4);
                        }
                    }

                    m1 = ty * W + tx;
                    m2 = m1 + W;
                    m3 = m2 + W;
                    m4 = m3 + W;

                    _mm256_store_ps(out + m1, c1);
                    _mm256_store_ps(out + m2, c2);
                    _mm256_store_ps(out + m3, c3);
                    _mm256_store_ps(out + m4, c4);
                }
            }

            // remaining rows inside the tile
            for (; ty < tile_end_y; ty++) {

                for (int tx = ox; tx < tile_end_x; tx += 8) {

                    c1 = _mm256_setzero_ps();

                    for (int j = 0; j < K; j++) {

                        stride1_a = (ty + j) * in_stride;
                        stride2 = j * K;

                        for (int i = 0; i < K; i++) {

                            a1 = _mm256_load_ps(in + stride1_a + tx + i);
                            b =  _mm256_set1_ps(ker[stride2 + i]);
                            c1 = _mm256_fmadd_ps(a1, b, c1);
                        }
                    }

                    _mm256_store_ps(out + ty * W + tx,c1);
                }
            }
        }
    }
}