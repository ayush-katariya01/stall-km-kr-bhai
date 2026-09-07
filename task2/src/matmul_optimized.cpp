// matmul_optimized.cpp  STAGE 3: PUT IT ALL TOGETHER
//
// This is the graded function AND the kernel that gets injected into llama.cpp. Combine
// everything you have learned across the whole assignment  loop reordering, register
// blocking and unrolling (Task 1 / Stage 1 here), cache tiling and software prefetch
// (Stage 2)  and TUNE it to be as fast as you can. Your speedup over matmul_naive determines
// your score (see the tier table the harness prints), and this same function will power a
// real LLM inference via `make llama-demo`.

#include <immintrin.h>

#include "matmul.h"

inline float hsum256(__m256 x)
{
    __m128 low = _mm256_castps256_ps128(x);
    __m128 high = _mm256_extractf128_ps(x, 1);

    __m128 sum = _mm_add_ps(low, high);

    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);

    return _mm_cvtss_f32(sum);
}


void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K,
                      int lda, int ldb, int ldc)
{
    const int TILE = 64;

    for (int ii = 0; ii < M; ii += TILE) {

        for (int jj = 0; jj < N; jj += TILE) {

            int i_end = (ii + TILE < M) ? ii + TILE : M;
            int j_end = (jj + TILE < N) ? jj + TILE : N;

            int i = ii;

            for (; i + 1 < i_end; i += 2) {

                const float* a0 =
                    A + static_cast<long>(i) * lda;

                const float* a1 =
                    A + static_cast<long>(i + 1) * lda;

                int j = jj;

                for (; j + 3 < j_end; j += 4) {

                    const float* b0 =
                        B + static_cast<long>(j) * ldb;

                    const float* b1 =
                        B + static_cast<long>(j + 1) * ldb;

                    const float* b2 =
                        B + static_cast<long>(j + 2) * ldb;

                    const float* b3 =
                        B + static_cast<long>(j + 3) * ldb;

                    __m256 c00 = _mm256_setzero_ps();
                    __m256 c01 = _mm256_setzero_ps();
                    __m256 c02 = _mm256_setzero_ps();
                    __m256 c03 = _mm256_setzero_ps();

                    __m256 c10 = _mm256_setzero_ps();
                    __m256 c11 = _mm256_setzero_ps();
                    __m256 c12 = _mm256_setzero_ps();
                    __m256 c13 = _mm256_setzero_ps();

                    int p = 0;

                    for (; p + 31 < K; p += 32) {

                        __m256 av0 = _mm256_loadu_ps(a0 + p);
                        __m256 av1 = _mm256_loadu_ps(a1 + p);

                        __m256 bv0 = _mm256_loadu_ps(b0 + p);
                        __m256 bv1 = _mm256_loadu_ps(b1 + p);
                        __m256 bv2 = _mm256_loadu_ps(b2 + p);
                        __m256 bv3 = _mm256_loadu_ps(b3 + p);

                        c00 = _mm256_fmadd_ps(av0, bv0, c00);
                        c01 = _mm256_fmadd_ps(av0, bv1, c01);
                        c02 = _mm256_fmadd_ps(av0, bv2, c02);
                        c03 = _mm256_fmadd_ps(av0, bv3, c03);

                        c10 = _mm256_fmadd_ps(av1, bv0, c10);
                        c11 = _mm256_fmadd_ps(av1, bv1, c11);
                        c12 = _mm256_fmadd_ps(av1, bv2, c12);
                        c13 = _mm256_fmadd_ps(av1, bv3, c13);


                        av0 = _mm256_loadu_ps(a0 + p + 8);
                        av1 = _mm256_loadu_ps(a1 + p + 8);

                        bv0 = _mm256_loadu_ps(b0 + p + 8);
                        bv1 = _mm256_loadu_ps(b1 + p + 8);
                        bv2 = _mm256_loadu_ps(b2 + p + 8);
                        bv3 = _mm256_loadu_ps(b3 + p + 8);

                        c00 = _mm256_fmadd_ps(av0, bv0, c00);
                        c01 = _mm256_fmadd_ps(av0, bv1, c01);
                        c02 = _mm256_fmadd_ps(av0, bv2, c02);
                        c03 = _mm256_fmadd_ps(av0, bv3, c03);

                        c10 = _mm256_fmadd_ps(av1, bv0, c10);
                        c11 = _mm256_fmadd_ps(av1, bv1, c11);
                        c12 = _mm256_fmadd_ps(av1, bv2, c12);
                        c13 = _mm256_fmadd_ps(av1, bv3, c13);


                        av0 = _mm256_loadu_ps(a0 + p + 16);
                        av1 = _mm256_loadu_ps(a1 + p + 16);

                        bv0 = _mm256_loadu_ps(b0 + p + 16);
                        bv1 = _mm256_loadu_ps(b1 + p + 16);
                        bv2 = _mm256_loadu_ps(b2 + p + 16);
                        bv3 = _mm256_loadu_ps(b3 + p + 16);

                        c00 = _mm256_fmadd_ps(av0, bv0, c00);
                        c01 = _mm256_fmadd_ps(av0, bv1, c01);
                        c02 = _mm256_fmadd_ps(av0, bv2, c02);
                        c03 = _mm256_fmadd_ps(av0, bv3, c03);

                        c10 = _mm256_fmadd_ps(av1, bv0, c10);
                        c11 = _mm256_fmadd_ps(av1, bv1, c11);
                        c12 = _mm256_fmadd_ps(av1, bv2, c12);
                        c13 = _mm256_fmadd_ps(av1, bv3, c13);


                        av0 = _mm256_loadu_ps(a0 + p + 24);
                        av1 = _mm256_loadu_ps(a1 + p + 24);

                        bv0 = _mm256_loadu_ps(b0 + p + 24);
                        bv1 = _mm256_loadu_ps(b1 + p + 24);
                        bv2 = _mm256_loadu_ps(b2 + p + 24);
                        bv3 = _mm256_loadu_ps(b3 + p + 24);

                        c00 = _mm256_fmadd_ps(av0, bv0, c00);
                        c01 = _mm256_fmadd_ps(av0, bv1, c01);
                        c02 = _mm256_fmadd_ps(av0, bv2, c02);
                        c03 = _mm256_fmadd_ps(av0, bv3, c03);

                        c10 = _mm256_fmadd_ps(av1, bv0, c10);
                        c11 = _mm256_fmadd_ps(av1, bv1, c11);
                        c12 = _mm256_fmadd_ps(av1, bv2, c12);
                        c13 = _mm256_fmadd_ps(av1, bv3, c13);
                    }

                    for (; p + 7 < K; p += 8) {

                        __m256 av0 = _mm256_loadu_ps(a0 + p);
                        __m256 av1 = _mm256_loadu_ps(a1 + p);

                        __m256 bv0 = _mm256_loadu_ps(b0 + p);
                        __m256 bv1 = _mm256_loadu_ps(b1 + p);
                        __m256 bv2 = _mm256_loadu_ps(b2 + p);
                        __m256 bv3 = _mm256_loadu_ps(b3 + p);

                        c00 = _mm256_fmadd_ps(av0, bv0, c00);
                        c01 = _mm256_fmadd_ps(av0, bv1, c01);
                        c02 = _mm256_fmadd_ps(av0, bv2, c02);
                        c03 = _mm256_fmadd_ps(av0, bv3, c03);

                        c10 = _mm256_fmadd_ps(av1, bv0, c10);
                        c11 = _mm256_fmadd_ps(av1, bv1, c11);
                        c12 = _mm256_fmadd_ps(av1, bv2, c12);
                        c13 = _mm256_fmadd_ps(av1, bv3, c13);
                    }

                    float s00 = hsum256(c00);
                    float s01 = hsum256(c01);
                    float s02 = hsum256(c02);
                    float s03 = hsum256(c03);

                    float s10 = hsum256(c10);
                    float s11 = hsum256(c11);
                    float s12 = hsum256(c12);
                    float s13 = hsum256(c13);

                    
                    for (; p < K; ++p) {

                        s00 += a0[p] * b0[p];
                        s01 += a0[p] * b1[p];
                        s02 += a0[p] * b2[p];
                        s03 += a0[p] * b3[p];

                        s10 += a1[p] * b0[p];
                        s11 += a1[p] * b1[p];
                        s12 += a1[p] * b2[p];
                        s13 += a1[p] * b3[p];
                    }

                    
                    // Store the 2 x 4 result block
                    C[(long)i * ldc + j]     = s00;
                    C[(long)i * ldc + j + 1] = s01;
                    C[(long)i * ldc + j + 2] = s02;
                    C[(long)i * ldc + j + 3] = s03;

                    C[(long)(i + 1) * ldc + j]     = s10;
                    C[(long)(i + 1) * ldc + j + 1] = s11;
                    C[(long)(i + 1) * ldc + j + 2] = s12;
                    C[(long)(i + 1) * ldc + j + 3] = s13;
                }

                
                // remaining columns in this tile
                for (; j < j_end; ++j) {

                    const float* b = B + static_cast<long>(j) * ldb;

                    float sum0 = 0.0f;
                    float sum1 = 0.0f;

                    for (int p = 0; p < K; ++p) {
                        sum0 += a0[p] * b[p];
                        sum1 += a1[p] * b[p];
                    }

                    C[(long)i * ldc + j] = sum0;
                    C[(long)(i + 1) * ldc + j] = sum1;
                }
            }

            
            // remaining single row
            for (; i < i_end; ++i) {

                const float* a = A + static_cast<long>(i) * lda;

                for (int j = jj; j < j_end; ++j) {

                    const float* b = B + static_cast<long>(j) * ldb;

                    float acc = 0.0f;

                    int p = 0;

                    for (; p + 7 < K; p += 8) {

                        __m256 av = _mm256_loadu_ps(a + p);
                        __m256 bv = _mm256_loadu_ps(b + p);

                        __m256 prod =_mm256_mul_ps(av, bv);

                        acc += hsum256(prod);
                    }

                    for (; p < K; ++p)
                        acc += a[p] * b[p];

                    C[(long)i * ldc + j] = acc;
                }
            }
        }
    }
}