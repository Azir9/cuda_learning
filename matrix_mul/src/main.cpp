#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "matmul.cuh"

void cpuMatmul(const float *A, const float *B, float *C, int width)
{
    for (int i = 0; i < width; ++i)
        for (int j = 0; j < width; ++j)
        {
            float s = 0.0f;
            for (int k = 0; k < width; ++k)
                s += A[i * width + k] * B[k * width + j];
            C[i * width + j] = s;
        }
}

int main()
{
    const int width    = 512;
    const int blockSize = 16;
    const size_t bytes = width * width * sizeof(float);

    std::vector<float> h_A(width * width);
    std::vector<float> h_B(width * width);
    std::vector<float> h_C(width * width);
    std::vector<float> ref(width * width);

    for (size_t i = 0; i < h_A.size(); ++i)
    {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // GPU
    matmulCuda(h_A.data(), h_B.data(), h_C.data(), width, blockSize);

    // CPU
    cpuMatmul(h_A.data(), h_B.data(), ref.data(), width);

    // 比较
    float err = 0.0f;
    for (size_t i = 0; i < h_C.size(); ++i)
        err = fmaxf(err, std::fabs(h_C[i] - ref[i]));
    printf("width=%d, blockSize=%d, max error = %e\n", width, blockSize, err);

    return 0;
}