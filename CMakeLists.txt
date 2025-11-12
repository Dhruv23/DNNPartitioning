cmake_minimum_required(VERSION 3.10)
project(sequential_inference_gpu LANGUAGES CXX)

find_package(CUDA REQUIRED)
include_directories(/usr/include /usr/local/TensorRT-8.6.1/include)
link_directories(/usr/lib/x86_64-linux-gnu /usr/local/TensorRT-8.6.1/lib)

add_executable(sequential_inference_gpu sequential_inference_gpu.cpp)
target_compile_features(sequential_inference_gpu PRIVATE cxx_std_17)
target_link_libraries(sequential_inference_gpu nvinfer cudart nvToolsExt)
