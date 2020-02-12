cmake_minimum_required(VERSION 3.3)
project(Example)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if (WIN32)
	add_definitions(-DUNICODE)
endif(WIN32)

set(INCLUDE_FILES
	AddProblemDataset.h
	CisLoader.h
	PackDataset.h
	PackDataset2.h
	PackDefinition.h
	RNN.h	
	CNN.h
	FFN.h
	IrisDataset.h
	IrisTest.h
	ContinuousTest.h
	SimpleContinuousEnv.h
	CartPole.h
	Maze.h
	MazeAction.h
	MazeExample.h
	MountainCar.h
	MotivationTest.h
	)

set(SOURCE_FILES 
	AddProblemDataset.cpp
	CisLoader.cpp
	PackDataset.cpp
	PackDataset2.cpp
	PackDefinition.cpp
	RNN.cpp	
	CNN.cpp
	FFN.cpp
	IrisDataset.cpp
	IrisTest.cpp
	ContinuousTest.cpp
	SimpleContinuousEnv.cpp
	CartPole.cpp
	Maze.cpp
	MazeAction.cpp
	MazeExample.cpp
	MountainCar.cpp
	MotivationTest.cpp
	main.cpp
	)

include_directories(${CLAB_DIR} ${CORE_DIR} ${RL_DIR})
add_executable(Example ${SOURCE_FILES} ${INCLUDE_FILES})
target_link_libraries(Example Coeus)