// conv_simd.cpp  STAGE 4: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "convolution.h"

void conv_simd(const float* in, float* out, const float* ker,
               int H, int W, int K) {

    // const int p = K / 2;
    // const int in_stride = W + 2 * p;  // padded row stride

    // for (int oy = 0; oy < H; ++oy) {
    //     for (int ox = 0; ox < W; ++ox) {
    //         float acc = 0.0f;
    //         for (int ky = 0; ky < K; ++ky) {
    //             for (int kx = 0; kx < K; ++kx) {
    //                 acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
    //             }
    //         }
    //         out[oy * W + ox] = acc;
    //     }
    // }

    __m128 a,b,c;
    int  m=0;
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride
    int stride1,stride2;
    
    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox < W; ox=ox+4) {
            c=_mm_setzero_ps();
            
            for(int j=0;j<K;j++){
            stride1=(oy + j) * in_stride;
            stride2=j * K;
            for(int i=0;i<K;i++){
                    a=_mm_load_ps(&in[stride1 + (ox + i)]);;
                    b=_mm_set1_ps(ker[stride2 + i]);
                    c=_mm_add_ps(c,_mm_mul_ps(a,b));
                }
            }
            _mm_store_ps(out+m, c);
            m=m+4;
        }
    }    
                
    // TODO(student): replace this placeholder with your AVX2 implementation.
    // conv_naive(in, out, ker, H, W, K);
}
