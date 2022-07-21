cmake_minimum_required(VERSION 3.5)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(GraphCutCuda LANGUAGES CXX CUDA)
enable_language(CUDA)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")


file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/*.cpp
                  ${PROJECT_SOURCE_DIR}/*.cu)
include_directories(${PROJECT_SOURCE_DIR})
add_executable(${PROJECT_NAME} ${SOURCES})
#set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# cuda
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures")
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_STANDARD 11)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --compiler-options -fno-strict-aliasing --ptxas-options=-v --generate-line-info")

include_directories(${CUDA_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})

#include_directories(/usr/local/cuda/include)
#target_link_libraries(${PROJECT_NAME} -L/usr/local/cuda/lib64 -lcudart -lcuda)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

