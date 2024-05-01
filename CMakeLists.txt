cmake_minimum_required(VERSION 3.17)
project(NVIDIA_SGEMM_PRACTICE)
enable_language(CUDA CXX)

# gcc/g++编译参数说明：
# -O1~3编译器优化选项的4个级别，-O1默认，级别越大优化效果越好，但编译时间越长;
# -std=c++11，采用C++11标准编译
set(CMAKE_CXX_FLAGS "-O3 -std=c++11")

# nvcc编译参数说明：
# -g:主机代码添加调试信息;
# -G:设备代码产生调试信息，将会禁用大多数编译器优化，造成设备代码运行缓慢;
# -Xptxas -dlcm=ca启用L1缓存，-Xptxas -dlcm=cg关闭L1缓存

# set(CUDA_NVCC_FLAGS -g;-G;-Xptxas;-dlcm=ca)
# set(CUDA_NVCC_FLAGS -Xptxas;-dlcm=cg)
find_package(CUDAToolkit)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80)
endif()
set(CUDA_NVCC_FLAGS -arch=compute_80;-code=compute_80)

# 若FIND CUDA ERROR，在~/.bashrc中添加配置环境变量和动态库路径
# CUDA_HOME=/usr/local/cuda
# export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 配置头文件搜索路径
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)
# 配置待编译的源文件路径
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC)
# 可执行文件输出路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
# 生成可执行文件
add_executable(sgemm sgemm.cu ${SRC})
# Enable gprof for the target for debug
# if (CMAKE_BUILD_TYPE MATCHES Debug)
#     message("test")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
#     set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler \"-g -pg\"")
#     set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")
# endif()
add_executable(check check.cu)

# link cudart cublas
target_link_libraries(sgemm ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY})

