#include "SingleBatchModule.h"

using namespace Coeus;

SingleBatchModule::SingleBatchModule(NeuralNetwork* p_network, NetworkGradient* p_network_gradient, IUpdateRule* p_update_rule, ICostFunction* p_cost_function, int p_batch) : IBatchModule(p_batch), 
	_network(p_network), 
	_cost_function(p_cost_function), 
	_network_gradient(p_network_gradient), 
	_update_rule(p_update_rule)
{
	_update_batch = p_network_gradient->get_empty_params();
}

SingleBatchModule::~SingleBatchModule()
= default;

double SingleBatchModule::run_batch(const int p_b, const int p_batch, vector<Tensor*>* p_input, vector<Tensor*>* p_target)
{
	double error = 0;

	for (auto it = _update_batch.begin(); it != _update_batch.end(); ++it) {
		_update_batch[it->first].fill(0);
	}

	for (int i = 0; i < p_batch; i++) {
		const int index = p_b * p_batch + i;
		_network->activate(p_input->at(index));
		error += _cost_function->cost(_network->get_output(), p_target->at(index));
		_network_gradient->calc_gradient(p_target->at(index));

		_update_rule->calc_update(_network_gradient->get_gradient());
		for (auto it = _update_rule->get_update()->begin(); it != _update_rule->get_update()->end(); ++it) {
			_update_batch[it->first] += it->second;
		}
	}

	return error;
}
