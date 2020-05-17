cmake_minimum_required(VERSION 3.8)

project(CNPY)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(${COEUS_DIR} ${ZLIB_INCLUDE_DIRS})

set(INCLUDE_FILES
	cnpy.h
	)

set(SOURCE_FILES
	cnpy.cpp
	)

add_library(CNPY STATIC ${SOURCE_FILES} ${INCLUDE_FILES})
target_link_libraries(CNPY ${ZLIB_LIBRARIES})