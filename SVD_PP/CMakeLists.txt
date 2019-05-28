cmake_minimum_required(VERSION 3.10)
project(svd)

set(CMAKE_CXX_STANDARD 11)

file(GLOB svd_src SVD.cpp)

add_executable(svd main.cpp ${svd_src})