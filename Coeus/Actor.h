#pragma once
#include "NeuralNetwork.h"
#include "BaseGradientAlgorithm.h"

namespace Coeus {

	class __declspec(dllexport) Actor
{
public:
	Actor(NeuralNetwork* p_network, BaseGradientAlgorithm* p_gradient_algorithm, double p_gamma, double p_beta = 1);
	~Actor();

	double train(Tensor* p_state0, int p_action, double p_value0, double p_value1, double p_reward);

protected:
	NeuralNetwork* _network;
	BaseGradientAlgorithm* _gradient_algorithm;
	double _beta;
	double _gamma;

	Tensor _target;
};

}

