// conv_simd.cpp  STAGE 4: SIMD with AVX2 intrinsics
#include <immintrin.h>
#include "stdio.h"

#include "convolution.h"

float sum256(__m256 v)
{
    __m128 low  = _mm256_castps256_ps128(v);
    __m128 high = _mm256_extractf128_ps(v, 1);

    __m128 sum = _mm_add_ps(low, high);

    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);

    return _mm_cvtss_f32(sum);
}

void conv_simd(const float* in, float* out, const float* ker,
               int H, int W, int K) {

    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride

    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox < W; ++ox) {
            float acc = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                }
            }
            out[oy * W + ox] = acc;
        }
    }

    int k2=K*K;
    int pairdone=0;

    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox < W; ++ox) {
            float acc = 0.0f;
            
            float temp[k2];
            int ii=0;
            for (int ky = 0; ky < K; ++ky) {
                int l=(oy + ky) * in_stride;
                for (int kx = 0; kx < K; ++kx) {
                    
                    temp[ii] =in[l+ (ox + kx)];
                    ii++;
                }
            }

            // for(int i=0;i<k2;){
            //     int next=(oy+)
            //     for(int j=0;j<K;j++,i=i+3){
            //         temp[i]=in[oy + j];
            //     }
            // }
            __m256 c=_mm256_setzero_ps();
            float m,n,p=0;
            for(int i=0;i<k2;){
                if((k2-i)>=8){
                    __m256 a=_mm256_loadu_ps(ker+i);
                    __m256 b=_mm256_loadu_ps(temp+i);
                    c=_mm256_add_ps(c,_mm256_mul_ps(a,b));
                    i=i+8;
                }else{
                    m=*(ker+i);
                    n=*(temp+i);
                    p=m*n;
                    i++;
                }

            }
            // for (int ky = 0; ky < K; ++ky) {
            //     for (int kx = 0; kx < K; ++kx) {
            //         _m256d a=_mm256_loadu_pd();
            //         _m256d b=_mm256_loadu_pd();
                    
            //         acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
            //     }
            // }

            p=p+sum256(c);


            out[oy * W + ox] = p;
        }
    }
    




    // TODO(student): replace this placeholder with your AVX2 implementation.
    // conv_naive(in, out, ker, H, W, K);
}
