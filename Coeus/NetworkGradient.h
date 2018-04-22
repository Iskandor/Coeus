#pragma once

#include "NeuralNetwork.h"
#include "ICostFunction.h"

namespace Coeus {

class __declspec(dllexport) NetworkGradient
{

public:
	NetworkGradient(NeuralNetwork* p_network, ICostFunction* p_cost_function);
	~NetworkGradient();

	void calc_gradient(Tensor* p_target);
	map<string, Tensor>* get_gradient() { return &_gradient; }

private:
	NeuralNetwork*	_network;
	ICostFunction*	_cost_function;

	map<string, Tensor> _gradient;
};

}
