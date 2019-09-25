#pragma once
#include "gym.h"
#include "Tensor.h"

class CartPoleAgent
{
public:
	CartPoleAgent();
	~CartPoleAgent();

	void run(const boost::shared_ptr<Gym::Client>& p_client, const std::string& p_env_id, int p_episodes) const;

private:
	static void copy_state(vector<float> &p_observation, Tensor &p_state);

	static const int STATE = 4;
	static const int ACTION = 1;
};

