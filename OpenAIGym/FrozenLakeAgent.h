#pragma once
#include "NeuralNetwork.h"
#include "QLearning.h"
#include "gym.h"

using namespace Coeus;

class FrozenLakeAgent
{
public:
	FrozenLakeAgent();
	~FrozenLakeAgent();
	void run(const boost::shared_ptr<Gym::Client>& p_client, const std::string& p_env_id, const int p_episodes);

private:
	static const int STATE = 16;
	static const int ACTION = 4;

	NeuralNetwork	_network;
	QLearning		*_agent;
};

