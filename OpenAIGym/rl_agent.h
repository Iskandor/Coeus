#pragma once
#include "../FLAB/Tensor.h"
#include <vector>
#include "NeuralNetwork.h"
#include "QLearning.h"
#include "gym.h"
#include "ADAM.h"

using namespace FLAB;
using namespace Coeus;

class rl_agent
{
public:
	rl_agent();
	~rl_agent();
	void run(const boost::shared_ptr<Gym::Client>& p_client, const std::string& p_env_id, const int p_episodes);

private:
	static const int STATE = 16;
	static const int ACTION = 4;

	NeuralNetwork	_network;
	BaseGradientAlgorithm *_optimizer;
	QLearning		*_agent;


	static Tensor encode_state(vector<float> &p_sensors);
	static int choose_action(Tensor *p_input, double p_epsilon);
	static void binary_encoding(double p_value, Tensor* p_vector);

};

