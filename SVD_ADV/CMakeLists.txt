cmake_minimum_required(VERSION 3.10)
project(svd_adv)

set(CMAKE_CXX_STANDARD 11)

file(GLOB svd_adv_src SVD.cpp)

add_executable(svd_adv main.cpp ${svd_adv_src})