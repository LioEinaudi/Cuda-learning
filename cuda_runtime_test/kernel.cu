#include <cuda_runtime.h>

// 基础版矩阵乘法核函数：C = A * B
__global__ void matrixMulKernel(float *A, float *B, float *C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
    {
        float sum = 0.0f;
        for (int i = 0; i < N; i++)
        {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

void launchMatrixMul(float *A, float *B, float *C, int N)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);
    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
}