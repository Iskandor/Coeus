#pragma once
#include "NeuralNetwork.h"
#include "ActorRule.h"

namespace Coeus {

class __declspec(dllexport) Actor
{
public:
	Actor(NeuralNetwork* p_network, GRADIENT_RULE p_grad_rule, double p_alpha);
	~Actor();

	double train(Tensor* p_state0, int p_action, double p_td_error) const;

protected:
	NeuralNetwork* _network;
	NetworkGradient* _network_gradient;
	ActorRule* _update_rule;
};

}

