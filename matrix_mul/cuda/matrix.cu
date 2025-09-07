#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cassert>

void Matmul_imply(float *A_host, float *B_host, float *C_Host, int width, int blockSize);
extern "C" void matmulCuda(const float *A, const float *B, float *C,
                           int width, int blockSize)
{
    Matmul_imply(const_cast<float*>(A),
           const_cast<float*>(B),
           C,
           width,
           blockSize);
}

/*
       **
       **
       **    
        y
*****  **x

*/






__global__ void matmulkernel(float* A_device,float* B_device,float* C_device,int width){

    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    for (int k = 0; k < width; ++k)
        sum += A_device[x * width + k] * B_device[k * width + y];
        //这里索引为什么是这个
    C_device[x * width + y] = sum;
    
}

//summery 索引
/*A * B = C*/
void Matmul_imply(float *A_host ,float *B_host, float* C_Host,int width,int blockSize){
    /* set the size of matrix*/
    int size = width * width * sizeof(float);

    float *A_device;
    float *B_device;

    cudaMalloc(&A_device,size);
    cudaMalloc(&B_device,size);
    // memory copy
    cudaMemcpy(A_device,A_host,size,cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(B_device,B_host,size,cudaMemcpyKind::cudaMemcpyHostToDevice);
    // result copy
    float *C_device;

    cudaMalloc(&C_device,size); //create memory
    //kernel function

    dim3 blockDim(blockSize,blockSize);
    dim3 gridDim(width/blockSize,width/blockSize);
    matmulkernel <<<gridDim,blockDim>>> (A_device,B_device,C_device,width);

    //copy the result
    cudaMemcpy(C_Host, C_device, size, cudaMemcpyDeviceToHost);  

    //synchronize
    cudaDeviceSynchronize();
    
    cudaFree(C_device);
    cudaFree(B_device);
    cudaFree(A_device);
}
