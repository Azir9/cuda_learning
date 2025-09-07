#ifndef MATMUL_CUH
#define MATMUL_CUH

extern "C" void matmulCuda(const float *A, const float *B, float *C,
                           int width, int blockSize);

#endif