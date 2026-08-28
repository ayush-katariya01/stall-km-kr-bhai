// conv_tile.cpp  STAGE 3: CACHE TILING

#include "convolution.h"

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    // TODO(student): replace this placeholder with your tiled/blocked implementation.
    /*cout << K << endl;
    conv_naive(in, out, ker, H, W, K);*/
    const int p = K / 2;
    const int in_stride = W + 2 * p;
    int b = 160; // output tile size, so input tile size will be b + k -1
    for(int i=0;i< H;i=i+b){
        for(int j=0;j<W;j=j+b){
            //tile looping
            int input_tile_end_i = (i + b < H) ? (i + b) : H;
            int input_tile_end_j = (j + b < W) ? (j + b) : W;
            for(int li = i; li < input_tile_end_i; li++){
                for(int lj = j; lj < input_tile_end_j; lj++){
                    float sum = 0.0f;
                    for(int ki = 0; ki < K; ki++){
                        for(int kj = 0;kj < K; kj++){
                            sum += in[(li + ki) * in_stride + (lj + kj)] * ker[ki * K + kj];
                        }
                    }
                    out[li * W + lj] = sum;
                }
            }
        }
    }
}
