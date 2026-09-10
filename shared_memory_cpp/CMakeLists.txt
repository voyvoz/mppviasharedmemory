cmake_minimum_required(VERSION 3.10)
project(shm_cxl_cpp CXX)

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -g -O3")

# Define the executable
add_executable(shm_cxl_cpp src/main.cpp)
target_include_directories(shm_cxl_cpp PRIVATE include)

# perf-cpp integration
find_package(perf-cpp REQUIRED)
target_link_libraries(shm_cxl_cpp perf-cpp::perf-cpp)
