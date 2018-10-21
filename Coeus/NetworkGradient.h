#pragma once

#include "NeuralNetwork.h"
#include "ICostFunction.h"

namespace Coeus {

class __declspec(dllexport) NetworkGradient
{

public:
	NetworkGradient(NeuralNetwork* p_network);
	~NetworkGradient();

	void calc_gradient(Tensor* p_target);
	map<string, Tensor>* get_w_gradient() { return &_w_gradient; }
	map<string, Tensor>* get_b_gradient() { return &_b_gradient; }	
	void check_gradient(Tensor* p_input, Tensor* p_target);

	void init(ICostFunction* p_cost_function);

private:
	IGradientComponent* create_component(BaseLayer* p_layer) const;
	double check_estimate(Tensor* p_input, Tensor* p_target) const;

	NeuralNetwork*	_network;
	ICostFunction*	_cost_function;

	map<string, IGradientComponent*> _gradient_component;

	map<string, Tensor> _w_gradient;
	map<string, Tensor> _b_gradient;

};

}
