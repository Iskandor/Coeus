cmake_minimum_required(VERSION 3.8)
project(Example)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if (WIN32)
	add_definitions(-DUNICODE)
endif(WIN32)

set(INCLUDE_FILES
	FFN.h
	maze_experiment.h
	maze.h
	mountain_car.h
	mountain_car_experiment.h
	simple_continuous_env.h
	)

set(SOURCE_FILES
	FFN.cpp
	main.cpp
	maze.cpp
	maze_experiment.cpp
	mountain_car.cpp
	mountain_car_experiment.cpp
	simple_continuous_env.cpp
	)

include_directories(${CUCLAB_DIR} ${CLAB_DIR} ${CORE_DIR} ${RL_DIR})
add_executable(Example ${SOURCE_FILES} ${INCLUDE_FILES})
target_link_libraries(Example Coeus)