cmake_minimum_required(VERSION 3.8)
project(CLAB VERSION 1.0.0)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(${MKL_INCLUDE_DIR})

set(INCLUDE_FILES
	../coeus.h
	activation_functions.h
	CLAB.h
	exponential_interpolation.h
	igate.h
	iinterpolation.h
	linear_interpolation.h
	loss_functions.h
	metrics.h
	ounoise.h
	random_generator.h
	tensor.h
	tensor_initializer.h
	tensor_operator_cpu.h
	)

set(SOURCE_FILES 
	activation_functions.cpp
	exponential_interpolation.cpp
	linear_interpolation.cpp
	loss_functions.cpp
	metrics.cpp
	ounoise.cpp
	random_generator.cpp
	tensor.cpp
	tensor_initializer.cpp
	tensor_operator_cpu.cpp
	)
	
include_directories(
	${COEUS_DIR}
	${ZLIB_INCLUDE_DIRS}
	${CNPY_DIR}
	${CUCLAB_DIR}
	)	
	
add_library(CLAB STATIC ${SOURCE_FILES} ${INCLUDE_FILES})
target_compile_options(CLAB PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX>)