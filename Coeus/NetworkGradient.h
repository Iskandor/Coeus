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
	map<string, Tensor>* get_w_gradient() { return &_w_gradient; }
	map<string, Tensor>* get_b_gradient() { return &_b_gradient; }	
	void update(map<string, Tensor> &p_update) const;

	void check_gradient(Tensor* p_input, Tensor* p_target);

private:
	
	double check_estimate(Tensor* p_input, Tensor* p_target) const;

	NeuralNetwork*	_network;
	ICostFunction*	_cost_function;

	map<string, Tensor> _w_gradient;
	map<string, Tensor> _b_gradient;

};

}
