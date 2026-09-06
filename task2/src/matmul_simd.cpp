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

    __m256 s,t1,t2,t3,t4,u1,u2,u3,u4;
    int m=0;
    for (int i = 0; i < M; ++i) {
        const float* a=A+i*K;
        for (int j = 0; j < N; j=j+2) {
            u1=_mm256_setzero_ps();
            u2=_mm256_setzero_ps();
            u3=_mm256_setzero_ps();
            u4=_mm256_setzero_ps();
            float acc1 =0;
            float acc2 =0;
            float acc3 =0;
            float acc4 =0;
            // const float* a = A + static_cast<long>(i) * lda;
            // const float* b = B + static_cast<long>(j) * ldb;

            const float* b1=B+j*K;
            const float* b2=B+(j+1)*K;
            const float* b3=B+(j+2)*K;
            const float* b4=B+(j+3)*K;
            for (int p = 0; p < K; p=p+8) {
                s=_mm256_load_ps(a+p);
                t1=_mm256_load_ps(b1+p);
                t2=_mm256_load_ps(b2+p);
                t3=_mm256_load_ps(b3+p);
                t4=_mm256_load_ps(b4+p);
                u1=_mm256_add_ps(u1,_mm256_mul_ps(s,t1));
                u2=_mm256_add_ps(u2,_mm256_mul_ps(s,t2));
                u3=_mm256_add_ps(u3,_mm256_mul_ps(s,t3));
                u4=_mm256_add_ps(u4,_mm256_mul_ps(s,t4));
            }

            float temp1[8];
            _mm256_store_ps(temp1, u1);
            for(int x=0;x<8;x++){
                acc1+=temp1[x];
            }
            C[m]=acc1;

            float temp2[8];
            _mm256_store_ps(temp2, u2);
            for(int x=0;x<8;x++){
                acc2+=temp2[x];
            }
            C[m+1]=acc2;

            float temp3[8];
            _mm256_store_ps(temp3, u3);
            for(int x=0;x<8;x++){
                acc3+=temp3[x];
            }
            C[m+2]=acc3;

            float temp4[8];
            _mm256_store_ps(temp4, u4);
            for(int x=0;x<8;x++){
                acc4+=temp4[x];
            }
            C[m+3]=acc4;

            m=m+2;
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
