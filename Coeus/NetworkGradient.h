#pragma once

#include "NeuralNetwork.h"
#include "ICostFunction.h"

namespace Coeus {

class __declspec(dllexport) NetworkGradient
{

public:
	NetworkGradient(NeuralNetwork* p_network);
	virtual ~NetworkGradient();

	void activate(Tensor* p_input);
	void activate(vector<Tensor*>* p_input);
	virtual void calc_gradient(Tensor* p_loss = nullptr);
	virtual void calc_gradient(vector<Tensor*>* p_input, Tensor* p_loss = nullptr);

	NeuralNetwork* get_network() const { return _network; }
	virtual map<string, Tensor>& get_gradient();
	map<string, Tensor>& get_regular_gradient() { return _gradient; }

	void reset();
	void set_recurrent_mode(RECURRENT_MODE p_value);
	RECURRENT_MODE get_recurrent_mode() const { return _recurrent_mode; }

protected:
	void calc_derivative();
	void unfold_layer(const string& p_layer);

	RECURRENT_MODE	_recurrent_mode;
	NeuralNetwork*	_network;

	map<string, Tensor>		_gradient;
	map<string, Tensor*>	_derivative;
	list<BaseLayer*> _calculation_graph;
};

}
