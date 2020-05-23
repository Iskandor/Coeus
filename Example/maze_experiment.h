#pragma once
#include "maze.h"
#include "neural_network.h"

class maze_experiment
{
public:
	maze_experiment();
	~maze_experiment();

	void run_qlearning(int p_episodes);
	void run_sarsa(int p_episodes);
	void run_dqn(int p_episodes);
	void run_ac(int p_episodes);

private:
	void test_agent(neural_network& p_agent) const;

	maze* _maze;
};

