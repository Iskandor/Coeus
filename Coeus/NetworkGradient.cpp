#include "NetworkGradient.h"
#include "NeuronOperator.h"
#include <queue>

using namespace Coeus;

NetworkGradient::NetworkGradient(NeuralNetwork* p_network)
{
	_network = p_network;
	_gradient.init(_network);
	_calculation_graph = _network->_backward_graph;
	_recurrent_mode = NONE;

	for(auto it : _network->_layers)
	{
		if (it.second->is_recurrent())
		{
			_recurrent_mode = BPTT;
		}
	}
}

NetworkGradient::~NetworkGradient()
{
	for (auto& it : _derivative)
	{
		delete it.second;
		it.second = nullptr;
	}
}

void NetworkGradient::calc_gradient(Tensor* p_loss) {

	BaseLayer* output_layer = _network->_layers[_network->_output_layer];
	output_layer->set_delta_out(p_loss);

	for (auto& it : _calculation_graph)
	{
		it->calc_gradient(_gradient, _derivative);
	}

	for(auto& it : _network->_input_layer)
	{
		_input_gradient[it] = _network->_layers[it]->get_delta_in(it);
	}
}

void NetworkGradient::calc_gradient(vector<Tensor*>* p_input, Tensor* p_loss)
{
	_calculation_graph.clear();

	string recurrent_node;

	for (auto& it : _network->_backward_graph)
	{
		if (it->is_recurrent())
		{
			if (recurrent_node.empty()) recurrent_node = it->get_id();
		}
		it->set_valid(false);
	}

	queue<string> q;
	q.push(_network->_output_layer);

	while (!q.empty())
	{
		const string v = q.front();
		q.pop();

		if (v == recurrent_node)
		{
			for(int i = 0; i < p_input->size(); i++)
			{
				unfold_layer(v);
			}
		}
		else
		{
			_calculation_graph.push_back(_network->get_layer(v));

			for (auto& it : _network->get_layer(v)->unfold_layer())
			{
				if (!_network->get_layer(it)->is_valid()) {
					q.push(it);
				}
			}
		}
	}

	calc_gradient(p_loss);
}

Gradient& NetworkGradient::get_gradient()
{
	return _gradient;
}

void NetworkGradient::reset()
{
	_network->reset();

	if (_recurrent_mode == BPTT)
	{
		_gradient.fill(0);
	}

	if (_recurrent_mode == RTRL)
	{
		for (auto& it : _derivative)
		{
			it.second->fill(0);
		}
	}
}

void NetworkGradient::set_recurrent_mode(const RECURRENT_MODE p_value)
{
	_recurrent_mode = p_value;
	for (const auto& it : _network->_layers)
	{
		it.second->set_mode(p_value);
	}
}

Tensor NetworkGradient::get_input_gradient(const int p_batch_size, const int p_column, const int p_size)
{
	Tensor result({p_batch_size, p_size}, Tensor::ZERO);
	Tensor::subregion(&result, _input_gradient[_network->_input_layer[0]], 0, p_column, p_batch_size, p_size);

	return result;
}

void NetworkGradient::calc_derivative()
{
	for (auto& it : _network->_backward_graph)
	{
		it->calc_derivative(_derivative);
	}
}

void NetworkGradient::unfold_layer(const string& p_layer)
{
	for (auto& it : _network->_backward_graph)
	{
		it->set_valid(false);
	}

	queue<string> q;
	q.push(p_layer);

	while (!q.empty())
	{
		const string v = q.front();
		q.pop();

		if (!_network->get_layer(v)->is_valid()) {
			_calculation_graph.push_back(_network->get_layer(v));
			_network->get_layer(v)->set_valid(true);

			for (auto& it : _network->get_layer(v)->unfold_layer())
			{
				if (!_network->get_layer(it)->is_valid()) {
					q.push(it);
				}
			}
		}
	}
}

void NetworkGradient::activate(Tensor* p_input)
{
	_network->_layers[_network->_input_layer[0]]->integrate(p_input);
	_network->activate();
	if (_recurrent_mode == RTRL) calc_derivative();
}

void NetworkGradient::activate(vector<Tensor*>* p_input)
{	
	reset();
	for (auto& it : *p_input)
	{
		_network->_layers[_network->_input_layer[0]]->integrate(it);
		_network->activate();		
		if (_recurrent_mode == RTRL) calc_derivative();
	}
}