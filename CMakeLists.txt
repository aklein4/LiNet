cmake_minimum_required(VERSION 3.19)
message(STATUS "CMake version: ${CMAKE_VERSION}")
enable_language("CUDA")

include(extern/DimensionalCUDA/CMakeLists.txt)

project(ChoiceNet)

include_directories(${PROJECT_SOURCE_DIR}/extern/DimensionalCUDA)
include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cu")

add_library(ChoiceNetLib ${cpu_source_files} ${gpu_source_files})
set_property(TARGET ChoiceNetLib PROPERTY CUDA_ARCHITECTURES "86")

add_executable(ChoiceNet "${CMAKE_SOURCE_DIR}/main.cpp")

target_link_libraries(ChoiceNet ChoiceNetLib)
target_link_libraries(ChoiceNet DimensionalCUDA)

add_definitions(-DUSE_CUDA)