// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + REGISTER BLOCKING + SOFTWARE PREFETCHING
//
// B is stored TRANSPOSED: the array passed in as `B` actually holds B^T,
// with shape N x K. Row j of the stored array (B + j*ldb) is therefore
// column j of the real, mathematical B matrix, contiguous in memory as
// K elements. This lets us compute
//   C[i][j] = dot(A_row_i, B^T_row_j) = sum_k A[i][k] * B[k][j]
// by walking both A's row i and B^T's row j with the same k index, which
// is what makes the SIMD dot-product kernels below valid.
//
// IMPORTANT: `ldb` must be the row stride of the TRANSPOSED storage, i.e.
// normally K (or K rounded up for alignment padding) -- NOT the original
// N. Passing ldb = N here will silently produce wrong results.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif

#ifndef PREFETCH_DISTANCE
#define PREFETCH_DISTANCE 64
#endif

#ifndef CACHE_FILL_LEVEL
#define CACHE_FILL_LEVEL 1
#endif

#include <immintrin.h>

#include "matmul.h"

namespace {

inline void prefetch_line(const float* p) {
#if CACHE_FILL_LEVEL == 1
    _mm_prefetch(reinterpret_cast<const char*>(p), _MM_HINT_T0);
#elif CACHE_FILL_LEVEL == 2
    _mm_prefetch(reinterpret_cast<const char*>(p), _MM_HINT_T1);
#elif CACHE_FILL_LEVEL == 3
    _mm_prefetch(reinterpret_cast<const char*>(p), _MM_HINT_T2);
#else
    _mm_prefetch(reinterpret_cast<const char*>(p), _MM_HINT_T0);
#endif
}

// Fast horizontal sum of an __m256 (avoids the store-to-array-then-loop trick).
inline float hsum256(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow = _mm_add_ps(vlow, vhigh);              // 4 partial sums
    __m128 shuf  = _mm_movehdup_ps(vlow);        // duplicate high pairs
    __m128 sums  = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

// Single-row (no register blocking) dot product kernel used for M-remainder
// rows. Uses TWO independent accumulators to break the FMA dependency chain
// even in the fallback path.
inline float dot_kernel_1x(const float* a, const float* b1,
                            int start_idx_k, int end_idx_k) {
    __m256 v_sum0 = _mm256_setzero_ps();
    __m256 v_sum1 = _mm256_setzero_ps();
    int k = start_idx_k;
    for (; k + 15 < end_idx_k; k += 16) {
        if (k + PREFETCH_DISTANCE < end_idx_k) {
            prefetch_line(a + k + PREFETCH_DISTANCE);
            prefetch_line(b1 + k + PREFETCH_DISTANCE);
        }
        __m256 va0 = _mm256_loadu_ps(a + k);
        __m256 vb0 = _mm256_loadu_ps(b1 + k);
        v_sum0 = _mm256_fmadd_ps(va0, vb0, v_sum0);

        __m256 va1 = _mm256_loadu_ps(a + k + 8);
        __m256 vb1 = _mm256_loadu_ps(b1 + k + 8);
        v_sum1 = _mm256_fmadd_ps(va1, vb1, v_sum1);
    }
    for (; k + 7 < end_idx_k; k += 8) {
        __m256 va = _mm256_loadu_ps(a + k);
        __m256 vb = _mm256_loadu_ps(b1 + k);
        v_sum0 = _mm256_fmadd_ps(va, vb, v_sum0);
    }
    float sum = hsum256(v_sum0) + hsum256(v_sum1);
    for (; k < end_idx_k; k++) {
        sum += a[k] * b1[k];
    }
    return sum;
}

// Register-blocked micro-kernel: 4 rows of A reused against ONE row of B.
// Loading b1[k] once and reusing it for 4 independent FMA chains both
// amortizes the B load and breaks the latency-bound dependency chain that
// a single accumulator suffers from.
inline void dot_kernel_4x1(const float* a0, const float* a1,
                            const float* a2, const float* a3,
                            const float* b1,
                            int start_idx_k, int end_idx_k,
                            float& s0, float& s1, float& s2, float& s3) {
    __m256 v0 = _mm256_setzero_ps();
    __m256 v1 = _mm256_setzero_ps();
    __m256 v2 = _mm256_setzero_ps();
    __m256 v3 = _mm256_setzero_ps();

    int k = start_idx_k;
    for (; k + 7 < end_idx_k; k += 8) {
        if (k + PREFETCH_DISTANCE < end_idx_k) {
            prefetch_line(a0 + k + PREFETCH_DISTANCE);
            prefetch_line(a1 + k + PREFETCH_DISTANCE);
            prefetch_line(a2 + k + PREFETCH_DISTANCE);
            prefetch_line(a3 + k + PREFETCH_DISTANCE);
            prefetch_line(b1 + k + PREFETCH_DISTANCE);
        }
        __m256 vb = _mm256_loadu_ps(b1 + k);   // loaded once, reused 4x
        v0 = _mm256_fmadd_ps(_mm256_loadu_ps(a0 + k), vb, v0);
        v1 = _mm256_fmadd_ps(_mm256_loadu_ps(a1 + k), vb, v1);
        v2 = _mm256_fmadd_ps(_mm256_loadu_ps(a2 + k), vb, v2);
        v3 = _mm256_fmadd_ps(_mm256_loadu_ps(a3 + k), vb, v3);
    }

    s0 = hsum256(v0);
    s1 = hsum256(v1);
    s2 = hsum256(v2);
    s3 = hsum256(v3);

    for (; k < end_idx_k; k++) {
        float bk = b1[k];
        s0 += a0[k] * bk;
        s1 += a1[k] * bk;
        s2 += a2[k] * bk;
        s3 += a3[k] * bk;
    }
}

} // namespace

void matmul_prefetch(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc) {
    const int b = BLOCK_SIZE;

    // Loop order: ib -> kb -> jb. Blocking K in the middle (rather than
    // innermost) keeps a given A/B tile pair resident across the full j
    // sweep for that k-block, which reuses cache lines better than
    // finishing all of k for one (i,j) pair before moving on.
    for (int start_idx_i = 0; start_idx_i < M; start_idx_i += b) {
        int end_idx_i = (start_idx_i + b < M) ? (start_idx_i + b) : M;

        for (int start_idx_k = 0; start_idx_k < K; start_idx_k += b) {
            int end_idx_k = (start_idx_k + b < K) ? (start_idx_k + b) : K;
            bool first_k_block = (start_idx_k == 0);

            for (int start_idx_j = 0; start_idx_j < N; start_idx_j += b) {
                int end_idx_j = (start_idx_j + b < N) ? (start_idx_j + b) : N;

                int i = start_idx_i;

                // --- Register-blocked main path: 4 rows of A at a time ---
                for (; i + 3 < end_idx_i; i += 4) {
                    const float* a0 = A + (long)i * lda;
                    const float* a1 = a0 + lda;
                    const float* a2 = a1 + lda;
                    const float* a3 = a2 + lda;

                    float* c0 = C + (long)i * ldc;
                    float* c1 = c0 + ldc;
                    float* c2 = c1 + ldc;
                    float* c3 = c2 + ldc;

                    for (int j = start_idx_j; j < end_idx_j; j++) {
                        const float* b1 = B + (long)j * ldb;

                        prefetch_line(&c0[j]);
                        prefetch_line(&c1[j]);
                        prefetch_line(&c2[j]);
                        prefetch_line(&c3[j]);

                        float s0, s1, s2, s3;
                        dot_kernel_4x1(a0, a1, a2, a3, b1,
                                       start_idx_k, end_idx_k, s0, s1, s2, s3);

                        if (first_k_block) {
                            c0[j] = s0; c1[j] = s1; c2[j] = s2; c3[j] = s3;
                        } else {
                            c0[j] += s0; c1[j] += s1; c2[j] += s2; c3[j] += s3;
                        }
                    }
                }

                // --- Remainder rows (M % 4 != 0 within this tile) ---
                for (; i < end_idx_i; i++) {
                    const float* a = A + (long)i * lda;
                    float* crow = C + (long)i * ldc;
                    for (int j = start_idx_j; j < end_idx_j; j++) {
                        const float* b1 = B + (long)j * ldb;
                        float s = dot_kernel_1x(a, b1, start_idx_k, end_idx_k);
                        if (first_k_block) {
                            crow[j] = s;
                        } else {
                            crow[j] += s;
                        }
                    }
                }
            }
        }
    }
}
