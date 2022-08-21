
# cmake boiler plate
cmake_minimum_required(VERSION 3.19)
message(STATUS "CMake version: ${CMAKE_VERSION}")

# cuda setup
enable_language("CUDA")
if(NOT DEFINED ${CMAKE_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()
message(STATUS "CUDA architectures set to ${CMAKE_CUDA_ARCHITECTURES}")

# I am speed
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# handle sub-libraries
include(extern/DimensionalCUDA/CMakeLists.txt)

# create the project
project(ChoiceNet)

# handle source and include files
include_directories(${PROJECT_SOURCE_DIR}/extern/DimensionalCUDA)
include_directories(${PROJECT_SOURCE_DIR}/include)
file(GLOB cpu_source_files "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")
file(GLOB gpu_source_files "${CMAKE_CURRENT_LIST_DIR}/src/*.cu")

# create the library and executible
add_library(ChoiceNetLib ${cpu_source_files} ${gpu_source_files})
add_executable(CN_test "${CMAKE_CURRENT_LIST_DIR}/main.cu")

# link libraries
target_link_libraries(CN_test ChoiceNetLib)
target_link_libraries(CN_test DimensionalCUDALib)