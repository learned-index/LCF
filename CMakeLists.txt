cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(test)

# compile options
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 14)

# package
find_package(Torch REQUIRED PATHS "/data1/qrstu1/yty_new/libtorch")
include_directories(src)
add_executable(test example/test.cc)
target_link_libraries(test ${TORCH_LIBRARIES})