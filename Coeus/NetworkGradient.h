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
	void calc_gradient(Tensor* p_value = nullptr);
	map<string, Tensor>* get_gradient() { return &_gradient; }
	map<string, Tensor> get_empty_params() const;
	void check_gradient(Tensor* p_input, Tensor* p_target, ICostFunction* p_loss);

private:
	void reset();
	void calc_deriv_estimate();
	IGradientComponent* create_component(BaseLayer* p_layer) const;
	float check_estimate(Tensor* p_input, Tensor* p_target, ICostFunction* p_loss) const;

	NeuralNetwork*	_network;

	map<string, IGradientComponent*> _gradient_component;
	map<string, Tensor> _gradient;

};

}
