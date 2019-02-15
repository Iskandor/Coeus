#pragma once
#include "NeuralNetwork.h"
#include "ActorRule.h"
#include "GradientAlgorithm.h"

namespace Coeus {

class __declspec(dllexport) Actor
{
public:
	Actor(NeuralNetwork* p_network, GradientAlgorithm* p_gradient_algorithm, float p_beta);
	Actor(NeuralNetwork* p_network, GRADIENT_RULE p_grad_rule, float p_alpha);
	~Actor();

	float train(Tensor* p_state0, int p_action, float p_td_error) const;

protected:
	NeuralNetwork* _network;
	NetworkGradient* _network_gradient;
	ActorRule* _update_rule;
	GradientAlgorithm* _gradient_algorithm;
	float _beta;
};

}

