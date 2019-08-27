#include "GradientAlgorithm.h"
#include <chrono>
#include "TensorOperator.h"

using namespace Coeus;

GradientAlgorithm::GradientAlgorithm(NeuralNetwork* p_network)
{
	_network = p_network;
	_network_gradient = new NetworkGradient(_network);
	_cost_function = nullptr;	
	_update_rule = nullptr;
	_batch_gradient = _network->get_empty_params();
	_recurrent_mode = NONE;
}


GradientAlgorithm::~GradientAlgorithm()
{
	delete _update_rule;
	delete _cost_function;
	delete _network_gradient;
}

float GradientAlgorithm::train(Tensor* p_input, Tensor* p_target)
{
	float error = 0;
	float alpha = 0;

	if (_recurrent_mode != NONE) _network_gradient->set_recurrent_mode(_recurrent_mode);
	_network_gradient->activate(p_input);
	error = _cost_function->cost(_network->get_output(), p_target);
	Tensor dloss = _cost_function->cost_deriv(_network->get_output(), p_target);

	_network_gradient->calc_gradient(&dloss);	
	_update_rule->calc_update(_network_gradient->get_gradient(), alpha);
	_network->update(_update_rule->get_update());

	if (_recurrent_mode != NONE) _network_gradient->set_recurrent_mode(NONE);
	
	return error;
}

float GradientAlgorithm::train(vector<Tensor*>* p_input, Tensor* p_target) {

	float error = 0;
	float alpha = 0;

	if (_recurrent_mode != NONE) _network_gradient->set_recurrent_mode(_recurrent_mode);
	_network_gradient->activate(p_input);
	error = _cost_function->cost(_network->get_output(), p_target);
	Tensor dloss = _cost_function->cost_deriv(_network->get_output(), p_target);

	_network_gradient->calc_gradient(p_input, &dloss);
	_update_rule->calc_update(_network_gradient->get_gradient(), alpha);
	_network->update(_update_rule->get_update());

	if (_recurrent_mode != NONE) _network_gradient->set_recurrent_mode(NONE);

	return error;
}

float GradientAlgorithm::train(vector<Tensor*>* p_input, vector<Tensor*>* p_target, bool p_update)
{
	float error = 0;
	float alpha = 0;

	if (_recurrent_mode != NONE) _network_gradient->set_recurrent_mode(_recurrent_mode);
	_network_gradient->reset();
	for(int i = 0; i < p_input->size(); i++)
	{
		_network_gradient->activate((*p_input)[i]);
		if ((*p_target)[i] != nullptr)
		{
			Tensor dloss = _cost_function->cost_deriv(_network->get_output(), (*p_target)[i]);
			error += _cost_function->cost(_network->get_output(), (*p_target)[i]);
			_network_gradient->calc_gradient(&dloss);

			for(auto it = _network_gradient->get_gradient()->begin(); it != _network_gradient->get_gradient()->end(); ++it)
			{
				TensorOperator::instance().vv_add(it->second.arr(), _batch_gradient[it->first].arr(), _batch_gradient[it->first].arr(), it->second.size());
			}
		}
	}
	if (_recurrent_mode != NONE) _network_gradient->set_recurrent_mode(NONE);

	if (p_update)
	{
		_update_rule->calc_update(&_batch_gradient, alpha);
		_network->update(_update_rule->get_update());
		for (auto& it : _batch_gradient)
		{
			it.second.fill(0);
		}
	}

	return error;
}

void GradientAlgorithm::reset() const
{
	_network_gradient->reset();
}

void GradientAlgorithm::set_recurrent_mode(const RECURRENT_MODE p_value)
{
	_recurrent_mode = p_value;	
}

void GradientAlgorithm::init(ICostFunction* p_cost_function, IUpdateRule* p_update_rule)
{
	_cost_function = p_cost_function;
	_update_rule = p_update_rule;

	// check network structure and set default BPTT mode if there are recurrent layers
	if (_recurrent_mode == NONE)
	{
		_recurrent_mode = _network_gradient->get_recurrent_mode();
	}
}
