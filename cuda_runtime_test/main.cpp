#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// 必须匹配 kernel.cu 里的 extern "C"
extern "C" void launchMatrixMul(float* A, float* B, float* C, int N);

void matrixMulCPU(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    int N = 512; 
    size_t size = N * N * sizeof(float);
    std::vector<float> h_A(N * N, 1.0f);
    std::vector<float> h_B(N * N, 2.0f);
    std::vector<float> h_C(N * N, 0.0f);
    std::vector<float> cpu_res(N * N, 0.0f);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);

    auto start_gpu = std::chrono::high_resolution_clock::now();
    launchMatrixMul(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost);

    std::cout << "Starting CPU calculation (N=512)..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    matrixMulCPU(h_A, h_B, cpu_res, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> ms_gpu = end_gpu - start_gpu;
    std::chrono::duration<double, std::milli> ms_cpu = end_cpu - start_cpu;

    std::cout << "GPU Time: " << ms_gpu.count() << " ms" << std::endl;
    std::cout << "CPU Time: " << ms_cpu.count() << " ms" << std::endl;
    std::cout << "Speedup: " << ms_cpu.count() / ms_gpu.count() << "x" << std::endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
