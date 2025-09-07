cmake_minimum_required(VERSION 3.16)
project(MatrixMul LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 头文件路径
include_directories(${CMAKE_SOURCE_DIR}/include)

# CUDA 库（设备代码）
add_library(matmul_cuda STATIC
    cuda/matrix.cu
)
set_target_properties(matmul_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


add_executable(matrix_mul
    src/main.cpp
)
target_link_libraries(matrix_mul PRIVATE matmul_cuda)