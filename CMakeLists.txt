cmake_minimum_required(VERSION 3.8.1)
message(STATUS "CMake version: ${CMAKE_VERSION}")

enable_language("CUDA")

project(ChoiceNet)

file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cu")

include_directories(${PROJECT_SOURCE_DIR}/include)

add_library(cuda ${cpu_source_files} ${gpu_source_files})

add_executable(ChoiceNet "${CMAKE_SOURCE_DIR}/main.cpp")

target_link_libraries(ChoiceNet cuda)

add_definitions(-DUSE_CUDA)