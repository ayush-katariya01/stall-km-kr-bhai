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


    //start

    int k2=K*K;
    __m256 a,b,c;
    int  m=0;
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride

    for (int oy = 0; oy < H; ++oy) {
        if(oy!=H-1){
            for (int ox = 0; ox < W; ox=ox+8) {
                c=_mm256_setzero_ps();
                
                for(int j=0;j<K;j++){
                    int stride=(oy + j) * in_stride;
                    for(int i=0;i<K;i++){
                        a=_mm256_load_ps(&in[stride + (ox + i)]);;
                        b=_mm256_set1_ps(ker[i]);
                        c=_mm256_add_ps(c,_mm256_mul_ps(a,b));
                    }
                }
                
                _mm256_storeu_ps(out+m, c);
                if((W-ox)!=8){
                    m=m+8;
                    // out[oy * W + ox] = acc;
                }
                else{
                    m=m+K-1;
                }
            }
        }
        else{
            for (int ox = 0; ox < W; ox=ox+8) {
                c=_mm256_setzero_ps();
                
                for(int j=0;j<K;j++){
                    int stride=(oy + j) * in_stride;
                    if(j!=K-1){
                        for(int i=0;i<K;i++){
                            a=_mm256_load_ps(&in[stride + (ox + i)]);
                            b=_mm256_set1_ps(ker[i]);
                            c=_mm256_add_ps(c,_mm256_mul_ps(a,b));
                        }
                    }
                    else{
                        if((W-ox)>8){
                            for(int i=0;i<K;i++){
                                a=_mm256_load_ps(&in[stride + (ox + i)]);
                                b=_mm256_set1_ps(ker[i]);
                                c=_mm256_add_ps(c,_mm256_mul_ps(a,b));
                            }
                        }
                        else{
                            float jugad[(W-ox)+K-1];

                            for(int jugadIndex=0;jugadIndex<8;jugadIndex++){
                                jugad[jugadIndex]=in[stride + (ox + jugadIndex)];
                            }
                            for(int jugadIndex=8;jugadIndex<8+K-1;jugadIndex++){
                                jugad[jugadIndex]=in[stride + (ox + jugadIndex)];
                            }
                            for(int i=0;i<K;i++){
                                a=_mm256_load_ps(jugad+i);
                                b=_mm256_set1_ps(ker[i]);
                                c=_mm256_add_ps(c,_mm256_mul_ps(a,b));
                            }
                            _mm256_storeu_ps(jugad, c);

                            for(int i=0;m<H*W;m++)
                            out[m]=jugad[i++];
                        }
                    }
                }
                
                _mm256_storeu_ps(out+m, c);
                if((W-ox)!=8){
                    m=m+8;
                    // out[oy * W + ox] = acc;
                }
                else{
                    m=m+K-1;
                }
            }
        }
    }    
    
    //end

    // TODO(student): replace this placeholder with your AVX2 implementation.
    // conv_naive(in, out, ker, H, W, K);
}
