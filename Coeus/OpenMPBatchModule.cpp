#include "OpenMPBatchModule.h"
#include <chrono>
#include <omp.h>

using namespace Coeus;

OpenMPBatchModule::OpenMPBatchModule(NeuralNetwork* p_network, IUpdateRule* p_update_rule, ICostFunction* p_cost_function, int p_batch) : IBatchModule(p_batch),
	_cost_function(p_cost_function),
	_network(p_network),
	_update_rule(p_update_rule)
{
	_clone_network = new NeuralNetwork*[p_batch];
	_network_gradient = new NetworkGradient*[p_batch];
	//_clone_update_rule = new IUpdateRule*[p_batch];
	_error = new double[p_batch];

	for (int i = 0; i < p_batch; i++)
	{
		_clone_network[i] = p_network->clone();
		_network_gradient[i] = new NetworkGradient(_clone_network[i]);
		//_clone_update_rule[i] = _update_rule->clone(_network_gradient[i]);
	}

	_update_batch = _network_gradient[0]->get_empty_params();
}

OpenMPBatchModule::~OpenMPBatchModule()
{
	for (int i = 0; i < _batch_size; i++)
	{
		delete _clone_network[i];
		delete _network_gradient[i];
		//delete _clone_update_rule[i];
	}

	//delete _clone_update_rule;
	delete _network_gradient;
	delete _clone_network;
	delete _error;
}

double OpenMPBatchModule::run_batch(int p_b, int p_batch, vector<Tensor*>* p_input, vector<Tensor*>* p_target)
{
	double error = 0;

	auto start = chrono::high_resolution_clock::now();

	for (auto it = _update_batch.begin(); it != _update_batch.end(); ++it) {
		_update_batch[it->first].fill(0);
	}

	omp_lock_t writelock;
	omp_init_lock(&writelock);

	#pragma omp parallel for
	for (int i = 0; i < p_batch; i++) {
		const int index = p_b * p_batch + i;
		//_clone_update_rule[i]->override(_update_rule);
		_clone_network[i]->calc_partial_derivs(p_input->at(index));
		_error[i] = _cost_function->cost(_clone_network[i]->get_output(), p_target->at(i));
		Tensor dloss = _cost_function->cost_deriv(_clone_network[i]->get_output(), p_target->at(index));
		_network_gradient[i]->calc_gradient(&dloss);
		//_clone_update_rule[i]->calc_update(_network_gradient[i]->get_gradient());
		omp_set_lock(&writelock);
			_update_rule->calc_update(_network_gradient[i]->get_gradient());
			for (auto it = _update_rule->get_update()->begin(); it != _update_rule->get_update()->end(); ++it) {
				_update_batch[it->first] += it->second;
			}
		omp_unset_lock(&writelock);
	}

	//_update_rule->merge(_clone_update_rule, _batch_size);

	for (int i = 0; i < _batch_size; i++)
	{
		error += _error[i];
	}

	omp_destroy_lock(&writelock);

	auto end = chrono::high_resolution_clock::now();

	//cout << "Batch" << p_b << ": " << (end - start).count() * ((double)chrono::high_resolution_clock::period::num / chrono::high_resolution_clock::period::den) << endl;

	return error;
}
