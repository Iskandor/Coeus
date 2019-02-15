#include "SingleBatchModule.h"

using namespace Coeus;

SingleBatchModule::SingleBatchModule(NeuralNetwork* p_network, NetworkGradient* p_network_gradient, ICostFunction* p_cost_function, int p_batch) : IBatchModule(p_batch), 
	_network(p_network), 
	_cost_function(p_cost_function), 
	_network_gradient(p_network_gradient)
{
	_gradient = p_network_gradient->get_empty_params();
}

SingleBatchModule::~SingleBatchModule()
= default;

void SingleBatchModule::run_batch(const int p_b, const int p_batch, vector<Tensor*>* p_input, vector<Tensor*>* p_target)
{
	for (auto& it : _gradient)
	{
		_gradient[it.first].fill(0);
	}

	for (int i = 0; i < p_batch; i++) {
		const int index = p_b * p_batch + i;
		_network_gradient->activate(p_input->at(index));
		Tensor dloss = _cost_function->cost_deriv(_network->get_output(), p_target->at(index));
		_network_gradient->calc_gradient(&dloss);

		for (auto& it : *_network_gradient->get_gradient())
		{
			_gradient[it.first] += it.second;
		}
	}
}

float SingleBatchModule::get_error(vector<Tensor*>* p_input, vector<Tensor*>* p_target)
{
	float error = 0;
	for (unsigned int i = 0; i < p_input->size(); i++)
	{
		_network->activate((*p_input)[i]);
		error += _cost_function->cost(_network->get_output(), (*p_target)[i]);
	}

	return error;
}
