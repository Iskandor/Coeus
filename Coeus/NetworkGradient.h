#pragma once

#include "NeuralNetwork.h"
#include "ICostFunction.h"
#include "Gradient.h"

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
	virtual Gradient& get_gradient();
	Gradient& get_regular_gradient() { return _gradient; }

	void reset();
	void set_recurrent_mode(RECURRENT_MODE p_value);
	RECURRENT_MODE get_recurrent_mode() const { return _recurrent_mode; }
	Tensor get_input_gradient(int p_batch_size, int p_column, int p_size);

protected:
	void calc_derivative();
	void unfold_layer(const string& p_layer);

	RECURRENT_MODE	_recurrent_mode;
	NeuralNetwork*	_network;

	Gradient				_gradient;
	map<string, Tensor*>	_derivative;
	list<BaseLayer*>		_calculation_graph;
	map<string, Tensor*>	_input_gradient;

	int _batch_size;
};

}
