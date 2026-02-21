#include <cuda_runtime.h>
#include <cstdio>

__global__ void matrixMulKernel(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 使用 extern "C" 确保 C++ 能正确识别这个函数名
extern "C" {
    void launchMatrixMul(float* A, float* B, float* C, int N) {
        dim3 threadsPerBlock(16, 16);
        dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);
        matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    }
}
