#include "NetworkGradient.h"
#include "NeuronOperator.h"
#include <queue>

using namespace Coeus;

NetworkGradient::NetworkGradient(NeuralNetwork* p_network)
{
	_network = p_network;
	_gradient = _network->get_empty_params();
	_calculation_graph = _network->_backward_graph;
	_recurrent_mode = NONE;
}

NetworkGradient::~NetworkGradient()
{
	for (auto& it : _delta)
	{
		delete it.second;
		it.second = nullptr;
	}

	for (auto& it : _derivative)
	{
		delete it.second;
		it.second = nullptr;
	}
}

void NetworkGradient::calc_gradient(Tensor* p_value) {

	calc_loss(p_value);

	for (auto& it : _calculation_graph)
	{
		it->calc_gradient(_gradient, _delta, _derivative);
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

	calc_loss(p_loss);

	for (auto& it : _calculation_graph)
	{
		it->calc_gradient(_gradient, _delta, _derivative);
	}
}

void NetworkGradient::reset()
{
	_network->reset();

	if (_recurrent_mode == BPTT)
	{
		for (auto& it : _gradient)
		{
			it.second.fill(0);
		}
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

void NetworkGradient::calc_loss(Tensor* p_value)
{
	BaseLayer* output_layer = _network->_layers[_network->_output_layer];

	output_layer->set_delta_out(p_value);
	if (p_value != nullptr)
	{
		if (p_value->rank() == 1)
		{
			_delta[_network->_output_layer] = NeuronOperator::init_auxiliary_parameter(_delta[_network->_output_layer], 1, output_layer->get_dim());
		}
		if (p_value->rank() == 2)
		{
			_delta[_network->_output_layer] = NeuronOperator::init_auxiliary_parameter(_delta[_network->_output_layer], p_value->shape(0), output_layer->get_dim());
		}
		_delta[_network->_output_layer]->override(p_value);		
	}
	else
	{
		_delta[_network->_output_layer] = new Tensor(_network->get_output()->rank(), Tensor::copy_shape(_network->get_output()->rank(), _network->get_output()->shape()), Tensor::ONES);
	}
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