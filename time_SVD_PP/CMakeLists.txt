cmake_minimum_required(VERSION 3.10)
project(time_svd)

set(CMAKE_CXX_STANDARD 11)

file(GLOB time_svd_src time_svd_pp.cpp)

add_executable(svd main.cpp ${time_svd_src})