// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "matmul.h"

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {

    // for (int i = 0; i < M; ++i) {
    //     for (int j = 0; j < N; ++j) {
    //         float acc = 0.0f;
    //         const float* a = A + static_cast<long>(i) * lda;
    //         const float* b = B + static_cast<long>(j) * ldb;
    //         for (int p = 0; p < K; ++p) {
    //             acc += a[p] * b[p];
    //         }
    //         C[static_cast<long>(i) * ldc + j] = acc;
    //     }
    // }

    __m128 s,t,u;
    int m=0;
    for (int i = 0; i < M; ++i) {
        const float* a=A+i*K;
        for (int j = 0; j < N; j++) {
            u=_mm_setzero_ps();
            float acc =0;
            // const float* a = A + static_cast<long>(i) * lda;
            // const float* b = B + static_cast<long>(j) * ldb;

            const float* b=B+j*K;
            for (int p = 0; p < K; p=p+4) {
                s=_mm_load_ps(a+p);
                t=_mm_load_ps(b+p);
                u=_mm_add_ps(u,_mm_mul_ps(s,t));
            }
            float temp[4];
            _mm_store_ps(temp, u);
            for(int x=0;x<4;x++){
                acc+=temp[x];
            }
            C[m++]=acc;
        }
    }

    // __m256 s,t,u;
    // int m=0;
    // for (int i = 0; i < M; ++i) {
    //     const float* a=A+i*K;
    //     for (int j = 0; j < N; j=j+8) {
    //         u=_mm256_setzero_ps();
    //         float acc =0;
    //         // const float* a = A + static_cast<long>(i) * lda;
    //         // const float* b = B + static_cast<long>(j) * ldb;

    //         const float* b1=B+(j)*K;
    //         const float* b2=b1+K;
    //         const float* b3=b2+K;
    //         const float* b4=b3+K;
    //         const float* b5=b4+K;
    //         const float* b6=b5+K;
    //         const float* b7=b6+K;
    //         const float* b8=b7+K;
    //         for (int p = 0; p < K; p++) {
    //             s=_mm256_set1_ps(*(a+p));
    //             t=_mm256_setr_ps(b1[p],b2[p],b3[p],b4[p],b5[p],b6[p],b7[p],b8[p]);
    //             u=_mm256_add_ps(u,_mm256_mul_ps(s,t));
    //         }
    //         _mm256_store_ps(C+m, u);
    //         m=m+8;
    //     }
    // }

    // TODO(student): replace this placeholder with your register-tiled AVX2 implementation.
    // matmul_naive(A, B, C, M, N, K, lda, ldb, ldc);
}