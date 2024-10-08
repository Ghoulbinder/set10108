project(set10108-cw1 LANGUAGES CXX CUDA)
cmake_minimum_required(VERSION 3.17)
set(CMAKE_CXX_STANDARD 17)

include_directories(../contrib)

add_executable(cw1 main.cpp)
add_executable(cw1-cuda cw1.cu)
set_property(TARGET cw1-cuda PROPERTY CUDA_SEPARABLE_COMPILATION ON)