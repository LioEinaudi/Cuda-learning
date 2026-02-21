cmake_minimum_required(VERSION 3.18)
project(CUDAGEMM LANGUAGES CXX CUDA)

# 告诉编译器去哪里找 CUDA 头文件
include_directories(/usr/local/cuda/include)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
add_executable(gem_test main.cpp kernel.cu)
set_target_properties(gem_test PROPERTIES CUDA_ARCHITECTURES 89)