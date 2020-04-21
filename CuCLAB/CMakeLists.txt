cmake_minimum_required(VERSION 3.8)

find_package(CUDA)

project(CuCLAB VERSION 1.0.0 LANGUAGES CXX CUDA)

set(INCLUDE_FILES
	cuda_kernels.cuh
	tensor_gpu.cuh
	tensor_operator_gpu.cuh
	)

set(SOURCE_FILES 
	tensor_gpu.cu
	tensor_operator_gpu.cu
	)
	
CUDA_ADD_LIBRARY(CuCLAB STATIC ${SOURCE_FILES} ${INCLUDE_FILES})
target_link_libraries(CuCLAB ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

set_target_properties(CuCLAB PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(CuCLAB PROPERTIES CUDA_SEPARABLE_COMPILATION ON)