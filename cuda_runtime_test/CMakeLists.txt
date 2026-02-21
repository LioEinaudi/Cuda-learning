cmake_minimum_required(VERSION 3.18)
project(CUDAGEMM LANGUAGES CXX CUDA)

include_directories(/usr/local/cuda/include)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
add_executable(gem_test main.cpp kernel.cu)
set_target_properties(gem_test PROPERTIES CUDA_ARCHITECTURES 89)