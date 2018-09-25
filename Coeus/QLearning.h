#pragma once
#include "NeuralNetwork.h"
#include "BaseGradientAlgorithm.h"

namespace Coeus {

class __declspec(dllexport) QLearning
{
public:
	QLearning(NeuralNetwork* p_network, BaseGradientAlgorithm* p_gradient_algorithm, double p_gamma, double p_alpha = 1.0);
	virtual ~QLearning();

	double train(Tensor* p_state0, int p_action0, Tensor* p_state1, double p_reward);

private:
	double calc_max_qa(Tensor* p_state) const;	

	NeuralNetwork* _network;
	BaseGradientAlgorithm* _gradient_algorithm;
	double _alpha;
	double _gamma;

	Tensor _target;
};

}

