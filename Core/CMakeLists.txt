cmake_minimum_required(VERSION 3.8)
project(Core VERSION 1.0.0)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(INCLUDE_FILES
	adam.h
	base64.h
	dense_layer.h
	id_generator.h
	linear_operator.h
	neural_network.h
	optimizer.h
	param.h
	param_model.h
	radam.h
	sgd.h
	variable.h
	)

set(SOURCE_FILES
	adam.cpp
	dense_layer.cpp
	linear_operator.cpp
	neural_network.cpp
	optimizer.cpp
	param.cpp
	param_model.cpp
	radam.cpp
	sgd.cpp
	variable.cpp
	)
	
include_directories(
	${COEUS_DIR}
	${CNPY_DIR}
	${CUCLAB_DIR}
	${CLAB_DIR}
	)
	
add_library(Core STATIC ${SOURCE_FILES} ${INCLUDE_FILES})
target_compile_options(Core PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX>)