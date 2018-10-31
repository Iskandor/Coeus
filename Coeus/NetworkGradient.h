#pragma once

#include "NeuralNetwork.h"
#include "ICostFunction.h"

namespace Coeus {

class __declspec(dllexport) NetworkGradient
{

public:
	NetworkGradient(NeuralNetwork* p_network);
	~NetworkGradient();

	void calc_gradient(Tensor* p_input, Tensor* p_target);
	map<string, Tensor>* get_gradient() { return &_gradient; }
	void check_gradient(Tensor* p_input, Tensor* p_target);

	void init(ICostFunction* p_cost_function);
	map<string, Tensor> get_empty_params() const;

private:
	IGradientComponent* create_component(BaseLayer* p_layer) const;
	double check_estimate(Tensor* p_input, Tensor* p_target) const;

	NeuralNetwork*	_network;
	ICostFunction*	_cost_function;

	map<string, IGradientComponent*> _gradient_component;

	map<string, Tensor> _gradient;

};

}
