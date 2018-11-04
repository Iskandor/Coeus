#pragma once

#include "NeuralNetwork.h"
#include "ICostFunction.h"

namespace Coeus {

class __declspec(dllexport) NetworkGradient
{

public:
	NetworkGradient(NeuralNetwork* p_network);
	~NetworkGradient();

	void init(ICostFunction* p_cost_function);
	void calc_gradient(Tensor* p_input, Tensor* p_target);
	map<string, Tensor>* get_gradient() { return &_gradient; }
	map<string, Tensor> get_empty_params() const;
	void check_gradient(Tensor* p_input, Tensor* p_target);

private:
	IGradientComponent* create_component(BaseLayer* p_layer) const;
	double check_estimate(Tensor* p_input, Tensor* p_target) const;

	NeuralNetwork*	_network;
	ICostFunction*	_cost_function;

	map<string, IGradientComponent*> _gradient_component;
	map<string, Tensor> _gradient;

};

}
