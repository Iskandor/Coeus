cmake_minimum_required(VERSION 3.3)
project(CoeusRL VERSION 1.0.0)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(INCLUDE_FILES
	A2C.h
	A3C.h
	ActorCritic.h
	BoltzmanExploration.h
	BufferItems.h
	CACLA.h
	CountModule.h
	DDPG.h
	DeepQLearning.h
	DoubleQLearning.h
	EGreedyExploration.h
	GAE.h
	IActorCritic.h
	ICM.h
	IEnvironment.h
	IMotivationModule.h
	NAC.h
	PolicyGradient.h
	QLearning.h
	SARSA.h
	TD.h
	)

set(SOURCE_FILES 
	A2C.cpp
	A3C.cpp
	ActorCritic.cpp
	BoltzmanExploration.cpp
	CACLA.cpp
	CountModule.cpp
	DDPG.cpp
	DeepQLearning.cpp
	DoubleQLearning.cpp
	EGreedyExploration.cpp
	GAE.cpp
	IActorCritic.cpp
	ICM.cpp
	IMotivationModule.cpp
	NAC.cpp
	PolicyGradient.cpp
	QLearning.cpp
	SARSA.cpp
	TD.cpp
	)

include_directories(
	${CORE_DIR}
	${CLAB_DIR}
	)
	
add_library(CoeusRL STATIC ${SOURCE_FILES} ${INCLUDE_FILES})