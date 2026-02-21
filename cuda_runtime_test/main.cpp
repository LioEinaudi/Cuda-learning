#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// 声明 kernel.cu 中定义的函数
void launchMatrixMul(float *A, float *B, float *C, int N);

// CPU 版矩阵乘法（用于对比性能和验证正确性）
void matrixMulCPU(const std::vector<float> &A, const std::vector<float> &B, std::vector<float> &C, int N)
{
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float sum = 0;
            for (int k = 0; k < N; ++k)
            {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main()
{
    int N = 1024; // 矩阵大小 N*N
    size_t size = N * N * sizeof(float);

    // 1. 在 CPU (Host) 上分配内存并初始化
    std::vector<float> h_A(N * N, 1.0f);     // 矩阵 A 全是 1
    std::vector<float> h_B(N * N, 2.0f);     // 矩阵 B 全是 2
    std::vector<float> h_C(N * N, 0.0f);     // 存放 GPU 结果
    std::vector<float> cpu_res(N * N, 0.0f); // 存放 CPU 结果

    // 2. 在 GPU (Device) 上分配内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // 3. 将数据从 Host 拷贝到 Device
    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);

    // --- GPU 计时开始 ---
    auto start_gpu = std::chrono::high_resolution_clock::now();

    launchMatrixMul(d_A, d_B, d_C, N);
    cudaDeviceSynchronize(); // 必须同步，否则计时不准（因为GPU是异步的）

    auto end_gpu = std::chrono::high_resolution_clock::now();
    // --- GPU 计时结束 ---

    // 4. 将结果拷贝回 Host
    cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost);

    // --- CPU 计时开始 (可选，N=1024 可能会跑几秒钟) ---
    std::cout << "Starting CPU calculation..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    matrixMulCPU(h_A, h_B, cpu_res, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();

    // 5. 结果校验（确保你的 GPU 代码写对了）
    bool correct = true;
    for (int i = 0; i < N * N; i++)
    {
        if (abs(h_C[i] - cpu_res[i]) > 1e-5)
        {
            correct = false;
            break;
        }
    }

    // 6. 输出报告
    std::chrono::duration<double, std::milli> ms_gpu = end_gpu - start_gpu;
    std::chrono::duration<double, std::milli> ms_cpu = end_cpu - start_cpu;

    std::cout << "Matrix Size: " << N << "x" << N << std::endl;
    std::cout << "Result Correct: " << (correct ? "YES" : "NO") << std::endl;
    std::cout << "GPU Time: " << ms_gpu.count() << " ms" << std::endl;
    std::cout << "CPU Time: " << ms_cpu.count() << " ms" << std::endl;
    std::cout << "Speedup: " << ms_cpu.count() / ms_gpu.count() << "x" << std::endl;

    // 7. 释放资源
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}