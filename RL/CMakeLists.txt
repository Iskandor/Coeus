cmake_minimum_required(VERSION 3.8)
project(RL VERSION 1.0.0)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(INCLUDE_FILES
	AC.h
	CACLA.h
	continuous_exploration.h
	DDPG.h
	discrete_exploration.h
	DQN.h
	forward_model.h
	ienvironment.h
	policy_gradient.h
	QAC.h
	Qlearning.h
	replay_buffer.h
	SARSA.h
	TD.h
	metacritic.h
	)

set(SOURCE_FILES
	AC.cpp
	CACLA.cpp
	continuous_exploration.cpp
	DDPG.cpp
	discrete_exploration.cpp
	DQN.cpp
	forward_model.cpp
	policy_gradient.cpp
	QAC.cpp
	Qlearning.cpp
	SARSA.cpp
	TD.cpp
	metacritic.cpp
	)
	
include_directories(
	${COEUS_DIR}
	${CUCLAB_DIR}
	${CNPY_DIR}
	${CLAB_DIR}
	${CORE_DIR}
	)
	
add_library(RL STATIC ${SOURCE_FILES} ${INCLUDE_FILES})
target_compile_options(RL PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX>)