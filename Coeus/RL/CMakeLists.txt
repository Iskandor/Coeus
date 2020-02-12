cmake_minimum_required(VERSION 3.3)
project(CoeusRL VERSION 1.0.0)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
	
	if (MSVC)
		add_compile_options(/openmp)
	endif()
endif()

set(INCLUDE_FILES
	A2C.h
	A3C.h
	ActorCritic.h
	BoltzmanExploration.h
	BufferItems.h
	CACER.h
	CACLA.h
	CountModule.h
	DDPG.h
	DeepQLearning.h
	DoubleQLearning.h
	EGreedyExploration.h
	ForwardModel.h
	GAE.h
	GM2.h
	IActor.h
	ICritic.h
	ICM.h
	IEnvironment.h
	IMotivationModule.h
	NAC.h
	PolicyGradient.h
	PPO.h
	QLearning.h
	SARSA.h
	TD.h
	)

set(SOURCE_FILES 
	A2C.cpp
	A3C.cpp
	ActorCritic.cpp
	BoltzmanExploration.cpp
	CACER.cpp
	CACLA.cpp
	CountModule.cpp
	DDPG.cpp
	DeepQLearning.cpp
	DoubleQLearning.cpp
	EGreedyExploration.cpp
	ForwardModel.cpp
	GAE.cpp
	GM2.cpp
	ICM.cpp
	IMotivationModule.cpp
	NAC.cpp
	PolicyGradient.cpp
	PPO.cpp
	QLearning.cpp
	SARSA.cpp
	TD.cpp
	)

include_directories(
	${CORE_DIR}
	${CLAB_DIR}
	)
	
add_library(CoeusRL STATIC ${SOURCE_FILES} ${INCLUDE_FILES})