// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include <immintrin.h>

#include "matmul.h"

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your cache-blocked SIMD + prefetch
    // implementation.
    //matmul_naive(A, B, C, M, N, K, lda, ldb, ldc);
    const int PREFETCH_DISTANCE = 16;
    //const int CACHE_FILL_LEVEL = 3;
    int b = 64;
    for(int start_idx_i = 0;start_idx_i < M;start_idx_i += b){
        for(int start_idx_j = 0;start_idx_j < N;start_idx_j += b){
            for(int start_idx_k = 0;start_idx_k < K;start_idx_k += b){
                //find end for tile loop
                int end_idx_i = (start_idx_i + b < M) ? (start_idx_i + b) : M;
                int end_idx_j = (start_idx_j + b < N) ? (start_idx_j + b) : N;
                int end_idx_k = (start_idx_k + b < K) ? (start_idx_k + b) : K;
                for(int i = start_idx_i; i < end_idx_i;i++){
                    const float *a = A + (long)i * lda;
                    for(int j = start_idx_j; j < end_idx_j; j++){
                        const float *b1 = B + (long)j * ldb;
                        /*float sum = (start_idx_k == 0) ? 0.0f : C[(long)i * ldc + j];
                        for(int k = start_idx_k; k < end_idx_k;k++){
                            if(k + PREFETCH_DISTANCE < end_idx_k){
                                _mm_prefetch((char *)(a + k + PREFETCH_DISTANCE),_MM_HINT_T0);
                                _mm_prefetch((char *)(b1 + k + PREFETCH_DISTANCE),_MM_HINT_T0);
                            }
                            sum += a[k]*b1[k];
                        }
                        C[(long)i * ldc + j] = sum;*/
                        int k;
                        __m256 v_sum = _mm256_setzero_ps();
                        for(int k = start_idx_k; k + 7 < end_idx_k;k += 8){
                            if(k + PREFETCH_DISTANCE < end_idx_k){
                                _mm_prefetch((char *)(a + k + PREFETCH_DISTANCE),_MM_HINT_T0);
                                _mm_prefetch((char *)(b1 + k + PREFETCH_DISTANCE),_MM_HINT_T0);
                            }
                            __m256 va = _mm256_loadu_ps(a + k);
                            __m256 vb = _mm256_loadu_ps(b1 + k);
                             v_sum = _mm256_fmadd_ps(va, vb, v_sum);
                        }
                        float sum_array[8];
                        _mm256_storeu_ps(sum_array, v_sum);
                        float tile_sum = 0;
                        for(int xi = 0;xi <8;xi++){
                            tile_sum += sum_array[xi];
                        }
                        for (; k < end_idx_k; k++) {
                            tile_sum += a[k] * b1[k];
                        }
                        if (start_idx_k == 0) {
                            C[(long)i * ldc + j] = tile_sum;
                        } 
                        else {
                            C[(long)i * ldc + j] += tile_sum;
                        }
                    }
                }
            }
        }
    }
}
