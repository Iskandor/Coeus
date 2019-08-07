#pragma once

#include "NeuralNetwork.h"
#include "ICostFunction.h"

namespace Coeus {

class __declspec(dllexport) NetworkGradient
{

public:
	NetworkGradient(NeuralNetwork* p_network);
	~NetworkGradient();

	void activate(Tensor* p_input);
	void activate(vector<Tensor*>* p_input);
	void calc_gradient(Tensor* p_value = nullptr);
	NeuralNetwork* get_network() const { return _network; }
	map<string, Tensor>* get_gradient() { return &_gradient; }
	
	void check_gradient(Tensor* p_input, Tensor* p_target, ICostFunction* p_loss);
	void reset();
	

private:
	void calc_loss(Tensor* p_value);
	void calc_derivative();
	float check_estimate(Tensor* p_input, Tensor* p_target, ICostFunction* p_loss) const;

	NeuralNetwork*	_network;

	map<string, Tensor>		_gradient;
	map<string, Tensor*>	_delta;
	map<string, Tensor*>	_derivative;
	
};

}
