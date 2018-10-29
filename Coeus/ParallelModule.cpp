#include "ParallelModule.h"
#include <ppl.h>

using namespace Coeus;
using namespace Concurrency;

ParallelModule::ParallelModule(NeuralNetwork* p_network, IUpdateRule* p_update_rule, ICostFunction* p_cost_function, const int p_batch) : 
	_batch(p_batch), _cost_function(p_cost_function), _network(p_network), _update_rule(p_update_rule)
{
	_clone_network = new NeuralNetwork*[p_batch];
	_network_gradient = new NetworkGradient*[p_batch];
	_clone_update_rule = new IUpdateRule*[p_batch];
	_error = new double[p_batch];

	for(int i = 0; i < p_batch; i++)
	{
		_clone_network[i] = p_network->clone();
		_network_gradient[i] = new NetworkGradient(_clone_network[i]);
		_network_gradient[i]->init(_cost_function);
		_clone_update_rule[i] = _update_rule->clone(_network_gradient[i]);
	}

	_update_batch = _network_gradient[0]->get_empty_params();
}


ParallelModule::~ParallelModule()
{
	for (int i = 0; i < _batch; i++)
	{
		delete _clone_network[i];
		delete _clone_update_rule[i];
		delete _network_gradient[i];
	}

	delete _clone_network;
	delete _clone_update_rule;
	delete _network_gradient;
	delete _error;
}

double ParallelModule::run_batch(int p_b, int p_batch, vector<Tensor*>* p_input, vector<Tensor*>* p_target)
{
	double error = 0;

	parallel_for(0, p_batch, [&](const int i) {
		const int index = p_b * p_batch + i;
		_clone_network[i]->activate(p_input->at(index));
		_error[i] = _cost_function->cost(_clone_network[i]->get_output(), p_target->at(i));
		
		_network_gradient[i]->calc_gradient(p_target->at(index));
		_clone_update_rule[i]->calc_update();
	});

	for (int i = 0; i < _batch; i++)
	{
		error += _error[i];
	}

	calc_update();

	return error;
}

void ParallelModule::calc_update()
{
	for (auto it = _update_batch.begin(); it != _update_batch.end(); ++it) {
		_update_batch[it->first].fill(0);
	}

	for (int i = 0; i < _batch; i++)
	{
		for (auto it = _clone_update_rule[i]->get_update()->begin(); it != _clone_update_rule[i]->get_update()->end(); ++it) {
			_update_batch[it->first] += it->second;
		}
	}

	for (auto it = _update_batch.begin(); it != _update_batch.end(); ++it) {
		_update_batch[it->first] /= _batch;
	}
}
