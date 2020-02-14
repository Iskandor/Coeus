cmake_minimum_required(VERSION 3.3)
project(CLAB VERSION 1.0.0)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(${MKL_INCLUDE_DIR})

set(INCLUDE_FILES
	IInterpolation.h
	ExponentialInterpolation.h
	LinearInterpolation.h
	Tensor.h
	TensorInitializer.h
	ITensorOperator.h
	TensorOperator.h
	TensorOperatorMKL.h
	TensorPool.h
	RandomGenerator.h
	OUNoise.h
	)

set(SOURCE_FILES 
	ExponentialInterpolation.cpp
	LinearInterpolation.cpp
	Tensor.cpp
	TensorInitializer.cpp
	TensorOperatorMKL.cpp
	TensorPool.cpp
	RandomGenerator.cpp
	OUNoise.cpp
	)
	
add_library(CLAB STATIC ${SOURCE_FILES} ${INCLUDE_FILES})