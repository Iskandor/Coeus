#include "PPLBatchModule.h"
#include <ppl.h>

using namespace Coeus;
using namespace Concurrency;

PPLBatchModule::PPLBatchModule(NeuralNetwork* p_network, IUpdateRule* p_update_rule, ICostFunction* p_cost_function, const int p_batch) : IBatchModule(p_batch),
	_cost_function(p_cost_function), 
	_network(p_network), 
	_update_rule(p_update_rule)
{
	_clone_network = new NeuralNetwork*[p_batch];
	_network_gradient = new NetworkGradient*[p_batch];
	_error = new double[p_batch];

	for(int i = 0; i < p_batch; i++)
	{
		_clone_network[i] = p_network->clone();
		_network_gradient[i] = new NetworkGradient(_clone_network[i]);
		_network_gradient[i]->init(_cost_function);
	}

	_update_batch = _network_gradient[0]->get_empty_params();
}


PPLBatchModule::~PPLBatchModule()
{
	for (int i = 0; i < _batch_size; i++)
	{
		delete _clone_network[i];
		delete _network_gradient[i];
	}

	delete _clone_network;
	delete _network_gradient;
	delete _error;
}

double PPLBatchModule::run_batch(int p_b, int p_batch, vector<Tensor*>* p_input, vector<Tensor*>* p_target)
{
	double error = 0;

	critical_section mutex;

	for (auto it = _update_batch.begin(); it != _update_batch.end(); ++it) {
		_update_batch[it->first].fill(0);
	}

	parallel_for(0, p_batch, [&](const int i) {
		const int index = p_b * p_batch + i;
		_clone_network[i]->activate(p_input->at(index));
		_error[i] = _cost_function->cost(_clone_network[i]->get_output(), p_target->at(i));
		
		_network_gradient[i]->calc_gradient(p_target->at(index));
		mutex.lock();
		_update_rule->calc_update(_network_gradient[i]->get_gradient());
		for (auto it = _update_rule->get_update()->begin(); it != _update_rule->get_update()->end(); ++it) {
			_update_batch[it->first] += it->second;
		}
		mutex.unlock();
	},
	static_partitioner()
	);

	for (int i = 0; i < _batch_size; i++)
	{
		error += _error[i];
	}

	return error;
}
