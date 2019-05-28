cmake_minimum_required(VERSION 3.10)
project(knn)

set(CMAKE_CXX_STANDARD 11)

file(GLOB knn_src knn.cpp)

add_executable(knn main.cpp ${knn_src})