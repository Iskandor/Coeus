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
	void calc_gradient(vector<Tensor*>* p_input, Tensor* p_loss = nullptr);

	NeuralNetwork* get_network() const { return _network; }
	map<string, Tensor>* get_gradient() { return &_gradient; }

	void reset();
	void set_recurrent_mode(RECURRENT_MODE p_value);

private:
	void calc_loss(Tensor* p_value);
	void calc_derivative();

	RECURRENT_MODE	_recurrent_mode;
	NeuralNetwork*	_network;

	map<string, Tensor>		_gradient;
	map<string, Tensor*>	_delta;
	map<string, Tensor*>	_derivative;
	list<BaseLayer*> _calculation_graph;
};

}
